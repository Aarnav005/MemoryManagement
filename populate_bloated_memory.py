import redis
import json
import numpy as np
from datetime import datetime, timedelta
import random
from sentence_transformers import SentenceTransformer

# --- NEW: Import redis-vl components to match the agent ---
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex

# Load the sentence-transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- MODIFIED: Setup redis-vl index exactly like the agent ---
REDIS_URL = "redis://localhost:6379"
# This client is still useful for clearing data and ZADD operations
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Define the EXACT SAME schema as your agent script
schema = IndexSchema.from_dict({
    "index": {
        "name": "message_index",
        "prefix": "message",
        "key_separator": ":"
    },
    "fields": [
        {"name": "content", "type": "text"},
        {
            "name": "embedding", # Field name must match agent's schema
            "type": "vector",
            "attrs": {
                "dims": 384,
                "algorithm": "FLAT",
                "distance_metric": "COSINE"
            }
        }
    ]
})

# Create the SearchIndex object to interact with the agent's index
redis_index = SearchIndex(schema=schema, redis_url=REDIS_URL)


def populate_compatible_memory():
    """Populate Redis with bloated data in a format compatible with the agent."""

    # Test Redis connection
    try:
        redis_client.ping()
        print("✅ Redis connection successful!")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please make sure Redis is running on localhost:6379")
        return

    # Clear existing index and data
    print("🧹 Clearing old data...")
    try:
        # Drop the index if it exists
        redis_index.drop(delete_documents=True)
        print("✅ Dropped existing index and documents")
    except Exception as e:
        print(f"No existing index to drop: {e}")
    
    # Create a fresh index
    print("🔧 Creating new 'message_index'...")
    redis_index.create(overwrite=True)
    print("✅ Created new index")

    # Note: Using KEYS in production is not recommended.
    print("🧹 Clearing old thread data...")
    keys = redis_client.keys("thread:*")
    if keys:
        redis_client.delete(*keys)

    # Sample Q&A organized by topic
    topics = {
        'baking': {
            'questions': [
                "How do I bake a chocolate cake?",
                "What's the secret to fluffy pancakes?",
                "How to make perfect chocolate chip cookies?",
                "What's the difference between baking powder and baking soda?",
                "How to prevent a cake from sinking in the middle?"
            ],
            'answers': [
                "For a moist chocolate cake, use oil instead of butter, add sour cream or yogurt, and don't overbake it. Brush with simple syrup after baking for extra moisture.",
                "For fluffy pancakes, avoid overmixing the batter and let it rest for 10-15 minutes before cooking. Use buttermilk and baking powder for extra lift.",
                "For perfect chocolate chip cookies, chill the dough for at least 30 minutes, use room temperature butter, and take them out when they look slightly underdone.",
                "Baking powder contains both an acid and a base, while baking soda needs an acidic ingredient to activate. Use 3x more baking powder than baking soda when substituting.",
                "To prevent sinking, ensure your cake is fully baked (use a toothpick test), don't open the oven door too early, and make sure your baking powder/soda is fresh."
            ]
        },
        'programming': {
            'questions': [
                "How do I learn Python programming?",
                "What's the difference between Python 2 and 3?",
                "How to optimize Python code?",
                "What are Python decorators?",
                "How to handle exceptions in Python?"
            ],
            'answers': [
                "Start with Python's official tutorial, practice with small projects, and work through exercises on platforms like LeetCode or HackerRank.",
                "Python 3 is the current version with many improvements, while Python 2 reached end-of-life in 2020. Key differences include print function syntax and integer division.",
                "Optimize Python by using built-in functions, list comprehensions, proper data structures, and libraries like NumPy for numerical operations.",
                "Decorators are functions that modify the behavior of other functions. They're used for logging, access control, and more.",
                "Use try/except blocks to handle exceptions gracefully. Catch specific exceptions and use finally for cleanup code."
            ]
        },
        'science': {
            'questions': [
                "What is quantum mechanics?",
                "How does photosynthesis work?",
                "What is the theory of relativity?",
                "How do black holes form?",
                "What is the structure of an atom?"
            ],
            'answers': [
                "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
                "Photosynthesis is the process by which green plants use sunlight to synthesize foods with carbon dioxide and water, producing oxygen as a byproduct.",
                "The theory of relativity, developed by Einstein, describes the laws of physics in different reference frames and includes the famous equation E=mc².",
                "Black holes form when massive stars collapse under their own gravity at the end of their life cycle, creating a region of space where the gravitational pull is extremely strong.",
                "An atom consists of a nucleus (protons and neutrons) surrounded by electrons in orbitals. The number of protons determines the element."
            ]
        },
        'history': {
            'questions': [
                "What caused World War I?",
                "Who was Cleopatra?",
                "What was the Renaissance?",
                "What was the Industrial Revolution?",
                "Who was Genghis Khan?"
            ],
            'answers': [
                "World War I was caused by militarism, alliances, imperialism, and the assassination of Archduke Franz Ferdinand of Austria-Hungary in 1914.",
                "Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt, known for her intelligence, political acumen, and relationships with Roman leaders.",
                "The Renaissance was a period of cultural, artistic, political and economic rebirth in Europe from the 14th to the 17th century, marking the transition from the Middle Ages to modernity.",
                "The Industrial Revolution was a period of major industrialization from the 18th to 19th centuries that transformed rural societies into industrialized urban ones.",
                "Genghis Khan was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous empire in history after his death."
            ]
        },
        'technology': {
            'questions': [
                "How does blockchain work?",
                "What is artificial intelligence?",
                "How do self-driving cars work?",
                "What is 5G technology?",
                "How do quantum computers work?"
            ],
            'answers': [
                "Blockchain is a decentralized, distributed ledger that records transactions across many computers. Each block contains transaction data and a cryptographic hash of the previous block.",
                "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction.",
                "Self-driving cars use sensors (like cameras, radar, and LIDAR), machine learning, and AI to navigate and make decisions without human input.",
                "5G is the fifth generation of wireless technology, offering faster speeds, lower latency, and the ability to connect more devices simultaneously compared to 4G networks.",
                "Quantum computers use quantum bits (qubits) that can exist in multiple states at once, allowing them to perform many calculations simultaneously."
            ]
        }
    }
    
    # Flatten the questions and answers for compatibility with existing code
    sample_questions = []
    sample_answers = []
    
    for topic in topics.values():
        sample_questions.extend(topic['questions'])
        sample_answers.extend(topic['answers'])
    
    # Create conversation threads with questions and answers
    num_threads = 20
    messages_per_thread = 10
    total_messages = 0
    
    print(f"\n📝 Generating {num_threads} conversation threads with {messages_per_thread} messages each...")
    
    for thread_num in range(num_threads):
        thread_id = f"conversation_{random.randint(1000, 9999)}"
        print(f"\n🧵 Thread {thread_num + 1}/{num_threads} ({thread_id}):")
        
        # Select a random topic for this thread
        topic_name = random.choice(list(topics.keys()))
        topic = topics[topic_name]
        
        for msg_num in range(messages_per_thread):
            # Alternate between user and assistant messages
            if msg_num % 2 == 0:
                # Get a random question from the selected topic
                topic_questions = topic['questions']
                topic_answers = topic['answers']
                
                # Ensure we don't go out of bounds
                q_idx = random.randint(0, len(topic_questions) - 1)
                user_input = topic_questions[q_idx]
                
                # Get the corresponding answer from the same topic
                assistant_response = topic_answers[q_idx]
                
                # Generate embedding for the user input
                embedding = st_model.encode(user_input).astype(np.float32).tobytes()
                
                # Create message data with both user input and next assistant response
                message_data = {
                    'user_input': user_input,
                    'assistant_response': assistant_response,  # Include the assistant's response
                    'thread_id': thread_id,
                    'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                    'message_type': 'user',
                    'similarity': 1.0
                }
                
                # Store in Redis
                message_id = f"{thread_id}:{msg_num:03d}"
                redis_client.hset(
                    f"message:{message_id}",
                    mapping={
                        'content': json.dumps(message_data),
                        'embedding': embedding,
                        'thread_id': thread_id,
                        'checkpoint_type': 'user',
                        'timestamp': message_data['timestamp']
                    }
                )
                
                # Add to thread's message list
                redis_client.zadd(f"thread:{thread_id}:messages", {message_id: msg_num})
                
                print(f"  👤 User: {user_input[:60]}...")
                
            else:
                # For assistant responses, we'll store the response with the previous user message
                # to maintain context in the embedding
                if msg_num > 0:  # Ensure there's a previous user message
                    prev_msg_id = f"{thread_id}:{msg_num-1:03d}"
                    # Get the previous message content directly
                    content = redis_client.hget(f"message:{prev_msg_id}", 'content')
                    if content:
                        try:
                            # Parse the existing content
                            prev_data = json.loads(content)
                            # Get the current topic's answers
                            topic_answers = topic['answers']
                            # Use the same index we used for the question
                            if 'q_idx' in locals() and q_idx < len(topic_answers):
                                prev_data['assistant_response'] = topic_answers[q_idx]
                                # Update in Redis
                                redis_client.hset(
                                    f"message:{prev_msg_id}",
                                    'content',
                                    json.dumps(prev_data)
                                )
                                print(f"  🔄 Updated previous message with assistant response")
                        except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as e:
                            print(f"  ⚠️ Error processing message {prev_msg_id}: {e}")
                continue  # Skip creating a separate assistant message
                
            total_messages += 1
    
    print(f"\n✅ Successfully populated Redis with {total_messages} messages across {num_threads} threads!")
    print("\n🔍 You can now run your agent to test the enhanced memory retrieval.")
    
    # Verify index count
    try:
        info = redis_index.info()
        print(f"\n📊 Index stats:")
        print(f"- Documents indexed: {info.get('num_docs', 0)}")
        print(f"- Index size: {info.get('used_memory', 0) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"\n⚠️ Could not get index stats: {e}")
    
    return True
    # Expanded questions and answers with more variety on the same topics
    sample_questions = [
        # Python related
        "What is Python?",
        "How do I install Python?",
        "What are the key features of Python?",
        "Is Python good for beginners?",
        "What can you build with Python?",
        "How does Python handle memory management?",
        "What are Python decorators?",
        "Explain Python's GIL (Global Interpreter Lock).",
        
        # France/Capital related
        "What is the capital of France?",
        "Tell me about Paris, France.",
        "What language is spoken in France?",
        "What are some famous landmarks in Paris?",
        "What's the population of France?",
        "What's the weather like in Paris?",
        "What's the best time to visit France?",
        "What's the currency used in France?",
        
        # Physics related
        "Explain the theory of relativity.",
        "What's special about Einstein's theory of relativity?",
        "How does general relativity differ from special relativity?",
        "What are some practical applications of relativity?",
        "How did Einstein come up with the theory of relativity?",
        "What is time dilation in relativity?",
        "How does gravity work in general relativity?",
        "What's the speed of light got to do with relativity?",
        
        # Cooking related
        "How do I bake a chocolate cake?",
        "What's the best chocolate cake recipe?",
        "How to make a moist chocolate cake?",
        "What are common mistakes when baking a cake?",
        "How to decorate a chocolate cake?",
        "What's the difference between baking powder and baking soda?",
        "How to make chocolate frosting for cakes?",
        "What's a good egg substitute for baking?",
        
        # Meditation related
        "What are the benefits of meditation?",
        "How to start meditating for beginners?",
        "What are different types of meditation?",
        "How long should I meditate each day?",
        "What's the best time to meditate?",
        "How to stay consistent with meditation?",
        "What are some common meditation techniques?",
        "Can meditation help with anxiety?",
        
        # Math related
        "How do I solve a quadratic equation?",
        "What's the quadratic formula?",
        "How to factor quadratic equations?",
        "What are the real-world applications of quadratic equations?",
        "How to graph a quadratic function?",
        "What's the discriminant in a quadratic equation?",
        "How to complete the square?",
        "What's the difference between linear and quadratic equations?",
        
        # Stock market related
        "What is the stock market?",
        "How does the stock market work?",
        "What are stocks and shares?",
        "How do I start investing in stocks?",
        "What's the difference between stocks and bonds?",
        "How to analyze a company's stock?",
        "What are dividends?",
        "What causes stock prices to change?",
        
        # Public speaking related
        "How do I improve my public speaking skills?",
        "What are tips for overcoming stage fright?",
        "How to structure an effective presentation?",
        "What are common public speaking mistakes?",
        "How to engage your audience when speaking?",
        "What's the importance of body language in public speaking?",
        "How to handle questions during a presentation?",
        "What are some good public speaking exercises?",
        
        # Web protocols related
        "What is the difference between HTTP and HTTPS?",
        "How does HTTPS work?",
        "Why is HTTPS important for websites?",
        "What is SSL/TLS encryption?",
        "How to enable HTTPS on a website?",
        "What are the risks of using HTTP?",
        "How does browser encryption work?",
        "What's the difference between HTTP/1.1 and HTTP/2?"
    ]
    
    sample_answers = [
        # Python answers
        "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "You can install Python by downloading it from python.org and following the installation instructions for your OS.",
        "Python's key features include easy-to-read syntax, dynamic typing, automatic memory management, and a large standard library.",
        "Yes, Python is excellent for beginners due to its simple and readable syntax, making it easier to learn programming concepts.",
        "With Python, you can build web applications, data analysis tools, machine learning models, automation scripts, and much more.",
        "Python uses automatic memory management with a private heap space and reference counting for garbage collection.",
        "Python decorators are functions that modify the behavior of other functions or methods without changing their source code.",
        "Python's GIL is a mutex that allows only one thread to execute Python bytecode at a time, which can impact multi-threaded performance.",
        
        # France answers
        "The capital of France is Paris.",
        "Paris is the capital and most populous city of France, known for its art, fashion, and landmarks like the Eiffel Tower and Louvre Museum.",
        "The official language of France is French, which is spoken by the majority of the population.",
        "Famous Paris landmarks include the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and Champs-Élysées.",
        "As of 2023, France has a population of approximately 68 million people.",
        "Paris has a temperate oceanic climate with mild winters (average 5°C/41°F) and warm summers (average 20°C/68°F).",
        "The best time to visit France is during spring (April-June) or fall (September-October) for pleasant weather and fewer crowds.",
        "France uses the Euro (€) as its official currency, which is also used by many other European countries.",
        
        # Physics answers
        "The theory of relativity, developed by Einstein, describes the laws of physics in the presence of gravitational fields and at high velocities.",
        "Einstein's theory of relativity revolutionized physics by showing that space and time are relative and interconnected, forming a four-dimensional spacetime.",
        "Special relativity deals with objects moving at constant speeds, while general relativity includes acceleration and gravity's effect on spacetime.",
        "Relativity has practical applications in GPS technology, particle accelerators, and understanding black holes and the universe's structure.",
        "Einstein developed relativity by questioning Newtonian physics and considering what would happen if he could travel at the speed of light.",
        "Time dilation is a phenomenon where time passes slower for objects moving at high speeds relative to an observer, as predicted by special relativity.",
        "In general relativity, gravity is not a force but rather the curvature of spacetime caused by mass and energy.",
        "The speed of light (about 300,000 km/s) is a fundamental constant in relativity, representing the maximum speed at which all energy and information can travel.",
        
        # Cooking answers
        "To bake a chocolate cake, mix flour, cocoa, sugar, eggs, and bake at 350°F for 30 minutes.",
        "The best chocolate cake recipe includes high-quality cocoa powder, buttermilk, and a bit of coffee to enhance the chocolate flavor.",
        "For a moist chocolate cake, use oil instead of butter, add sour cream or yogurt, and avoid overbaking.",
        "Common cake-baking mistakes include overmixing the batter, incorrect oven temperature, and opening the oven door too early.",
        "Decorate a chocolate cake with chocolate ganache, buttercream frosting, fresh berries, or chocolate shavings for an elegant finish.",
        "Baking powder contains both an acid and a base, while baking soda is pure sodium bicarbonate and requires an acid to activate.",
        "For chocolate frosting, melt chocolate with heavy cream for ganache, or mix butter, powdered sugar, and cocoa powder for buttercream.",
        "Good egg substitutes in baking include applesauce, mashed bananas, yogurt, or commercial egg replacers.",
        
        # Meditation answers
        "Meditation can reduce stress, improve focus, and promote emotional health.",
        "Start meditating by finding a quiet space, sitting comfortably, focusing on your breath, and gently bringing your attention back when it wanders.",
        "Different types include mindfulness, focused attention, loving-kindness, body scan, and transcendental meditation.",
        "Beginners can start with 5-10 minutes daily, gradually increasing to 20-30 minutes as they become more comfortable.",
        "Morning meditation can set a positive tone for the day, while evening meditation can help with relaxation and better sleep.",
        "Set a regular schedule, use guided meditations, and be patient with yourself to maintain consistency.",
        "Common techniques include breath awareness, body scanning, mantra repetition, and visualization.",
        "Yes, regular meditation can reduce anxiety by activating the parasympathetic nervous system and promoting relaxation.",
        
        # Math answers
        "A quadratic equation can be solved using the quadratic formula: x = (-b ± sqrt(b²-4ac)) / 2a.",
        "The quadratic formula is x = (-b ± sqrt(b²-4ac)) / 2a, which gives the solutions to any quadratic equation in the form ax² + bx + c = 0.",
        "To factor a quadratic equation, find two numbers that multiply to ac and add to b, then rewrite and factor by grouping.",
        "Quadratic equations model projectile motion, profit maximization, and area calculations in real-world scenarios.",
        "To graph a quadratic function, find the vertex, axis of symmetry, x-intercepts, and y-intercept, then plot these points.",
        "The discriminant (b²-4ac) determines the nature of the roots: positive for two real roots, zero for one real root, negative for complex roots.",
        "Completing the square involves rewriting a quadratic in the form (x-h)² + k, useful for finding the vertex and solving equations.",
        "Linear equations graph as straight lines with one solution, while quadratics graph as parabolas with up to two real solutions.",
        
        # Stock market answers
        "The stock market is a platform where shares of publicly held companies are bought and sold.",
        "The stock market works through exchanges where buyers and sellers trade shares based on supply and demand, influenced by company performance and economic factors.",
        "Stocks represent ownership in a company, while shares are the individual units of stock that can be bought or sold.",
        "Start investing by opening a brokerage account, researching companies, and considering index funds or ETFs for diversification.",
        "Stocks represent ownership in companies with potential for higher returns but more risk, while bonds are loans to governments or corporations with fixed returns.",
        "Analyze stocks by examining financial statements, valuation metrics (P/E ratio, etc.), industry position, and management quality.",
        "Dividends are payments made by companies to shareholders, typically from profits, usually paid quarterly.",
        "Stock prices change based on company performance, economic indicators, interest rates, investor sentiment, and market supply and demand.",
        
        # Public speaking answers
        "Practice, preparation, and feedback are key to improving public speaking skills.",
        "Overcome stage fright by practicing thoroughly, focusing on your message, using deep breathing, and visualizing success.",
        "Structure presentations with a clear introduction (tell them what you'll tell them), body (tell them), and conclusion (tell them what you told them).",
        "Common mistakes include reading slides verbatim, not preparing enough, ignoring the audience, and speaking too quickly.",
        "Engage your audience by asking questions, telling stories, using humor, making eye contact, and encouraging participation.",
        "Body language accounts for over 50% of communication impact, including posture, gestures, facial expressions, and movement.",
        "Handle questions by listening fully, repeating complex questions, being honest if you don't know, and keeping answers concise.",
        "Practice exercises include recording yourself, speaking in front of a mirror, joining Toastmasters, and doing tongue twisters.",
        
        # Web protocols answers
        "HTTPS is the secure version of HTTP, encrypting data between browser and server.",
        "HTTPS works by using SSL/TLS encryption to create a secure connection, verified by digital certificates from trusted authorities.",
        "HTTPS is crucial for protecting sensitive data, improving SEO rankings, and building user trust through browser security indicators.",
        "SSL/TLS (Secure Sockets Layer/Transport Layer Security) are cryptographic protocols that provide secure communication over a computer network.",
        "To enable HTTPS, obtain an SSL certificate from a certificate authority, install it on your web server, and configure your site to use HTTPS.",
        "Using HTTP risks data interception, man-in-the-middle attacks, and browser warnings that can drive away visitors.",
        "Browser encryption uses public-key cryptography to establish a secure connection and exchange a session key for encrypting data.",
        "HTTP/2 improves performance over HTTP/1.1 with features like multiplexing, header compression, and server push."
    ]

    conversations = []
    for i in range(80):
        user_input = sample_questions[i % len(sample_questions)]
        assistant_response = sample_answers[i % len(sample_answers)]
        bloated_metadata = {
            "session_id": f"sess_{10000+i}", "user_agent": "Mozilla/5.0 (TestAgent)", "ip_address": f"192.168.1.{100+i}",
            "version": "1.0.0", "debug_info": {"line_number": i},
            "performance_metrics": {"response_time_ms": 100+i},
            "analytics_data": {"page_view": True, "user_segment": "test"},
            "system_info": {"os": "TestOS", "browser": "TestBrowser"},
            "unnecessary_metadata_1": "This field is completely irrelevant",
            "unnecessary_metadata_2": "Another irrelevant field"
        }
        conversations.append((user_input, assistant_response, bloated_metadata))

    print(f"📝 Populating 'message_index' with {len(conversations)} bloated conversation messages...")

    base_time = datetime.now() - timedelta(days=30)

    for i, (user_input, assistant_response, bloated_metadata) in enumerate(conversations):
        thread_id = f"conversation_{i + 1}"
        timestamp = base_time + timedelta(days=random.randint(0, 29))
        message_id = f"{thread_id}:{timestamp.isoformat()}"

        # Prepare the core message data. The agent expects this to be in the 'content' field.
        # This JSON string is what the agent's LLM will later parse and filter.
        message_content_data = {
            'id': message_id,
            'timestamp': timestamp.isoformat(),
            'thread_id': thread_id,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'tools_used': [],
            **bloated_metadata
        }
        content_str = json.dumps(message_content_data, ensure_ascii=False, sort_keys=True)
        embedding_bytes = np.array(st_model.encode(content_str), dtype=np.float32).tobytes()

        # --- MODIFIED: Use redis_index.load() to populate the index ---
        data_to_load = {
            'id': message_id,
            'content': content_str,
            'embedding': embedding_bytes # Field name matches schema
        }

        try:
            redis_index.load([data_to_load], id_field="id")

            # --- MODIFIED: Use the correct key for short-term memory ---
            redis_client.zadd(f"thread:{thread_id}:messages", {message_id: timestamp.timestamp()})

        except Exception as e:
            print(f"[ERR] Failed to load message:{message_id} - {e}")

    print("\n✅ Successfully populated Redis in a format COMPATIBLE with the agent.")
    total_messages = len(redis_client.keys("message:*"))
    total_threads = len(redis_client.keys("thread:*"))
    print(f"   - {total_messages} messages loaded into the vector index.")
    print(f"   - {total_threads} conversation threads created for short-term memory.")
    print("\n🚀 You can now run your agent script. It will be able to retrieve and filter this data.")

if __name__ == "__main__":
    populate_compatible_memory()