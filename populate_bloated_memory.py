import redis
import json
import numpy as np
from datetime import datetime, timedelta
import hashlib
import random
# import google.generativeai as genai  # No longer needed for embeddings
import base64
from sentence_transformers import SentenceTransformer

# Load the sentence-transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Redis client for memory operations
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
# Binary Redis client for vector storage
redis_client_bin = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Configure Gemini (add your API key)
# genai.configure(api_key="AIzaSyCeDD6d7IZMm8e5_IbKJAIY_GiPx3BHXnU") # No longer needed for embeddings


def populate_bloated_memory():
    """Populate Redis with bloated conversation data containing extraneous metadata"""
    
    # Test Redis connection
    try:
        redis_client.ping()
        print("✅ Redis connection successful!")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please make sure Redis is running on localhost:6379")
        return
    
    # Clear existing data
    print("🧹 Clearing existing data...")
    keys = redis_client.keys("message:*")
    if keys:
        redis_client.delete(*keys)
    keys = redis_client.keys("embedding:*")
    if keys:
        redis_client.delete(*keys)
    keys = redis_client.keys("thread:*")
    if keys:
        redis_client.delete(*keys)
    
    # Sample conversations with BLOATED metadata
    sample_questions = [
        "What is Python?",
        "How do I install Python?",
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How do I bake a chocolate cake?",
        "What are the benefits of meditation?",
        "How do I solve a quadratic equation?",
        "What is the stock market?",
        "How do I improve my public speaking skills?",
        "What is the difference between HTTP and HTTPS?",
        "How do I start learning guitar?",
        "What is quantum computing?",
        "How do I write a resume?",
        "What is the best way to learn a new language?",
        "How do I change a flat tire?",
        "What is blockchain technology?",
        "How do I prepare for a job interview?",
        "What is the fastest land animal?",
        "How do I make a budget?",
        "What is the importance of sleep?",
        "How do I create a website?",
        "What is artificial intelligence?",
        "How do I train for a marathon?",
        "What is the Pythagorean theorem?",
        "How do I invest in stocks?",
        "What is the greenhouse effect?",
        "How do I organize my time better?",
        "What is the tallest mountain in the world?",
        "How do I cook pasta?",
        "What is the meaning of life?",
        "How do I reduce stress?",
        "What is the Fibonacci sequence?",
        "How do I set up a Wi-Fi network?",
        "What is the difference between RAM and ROM?",
        "How do I write a cover letter?",
        "What is the function of the heart?",
        "How do I learn to draw?",
        "What is the law of supply and demand?",
        "How do I clean my laptop keyboard?",
        "What is the speed of light?",
        "How do I meditate?",
        "What is the best way to memorize information?",
        "How do I plant a tree?",
        "What is the difference between a virus and bacteria?",
        "How do I make coffee?",
        "What is the purpose of government?",
        "How do I tie a tie?",
        "What is the largest ocean on Earth?",
        "How do I recycle properly?",
        "What is the process of photosynthesis?"
    ]
    sample_answers = [
        "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "You can install Python by downloading it from python.org and following the installation instructions for your OS.",
        "The capital of France is Paris.",
        "The theory of relativity, developed by Einstein, describes the laws of physics in the presence of gravitational fields.",
        "To bake a chocolate cake, mix flour, cocoa, sugar, eggs, and bake at 350°F for 30 minutes.",
        "Meditation can reduce stress, improve focus, and promote emotional health.",
        "A quadratic equation can be solved using the quadratic formula: x = (-b ± sqrt(b²-4ac)) / 2a.",
        "The stock market is a platform where shares of publicly held companies are bought and sold.",
        "Practice, preparation, and feedback are key to improving public speaking skills.",
        "HTTPS is the secure version of HTTP, encrypting data between browser and server.",
        "Start learning guitar by practicing basic chords and strumming patterns daily.",
        "Quantum computing uses quantum bits to perform computations much faster for certain problems.",
        "A resume should highlight your skills, experience, and education relevant to the job.",
        "Immersion, practice, and using language learning apps are effective ways to learn a new language.",
        "To change a flat tire, loosen the lug nuts, jack up the car, replace the tire, and tighten the nuts.",
        "Blockchain is a decentralized ledger technology used for secure and transparent transactions.",
        "Prepare for a job interview by researching the company and practicing common questions.",
        "The fastest land animal is the cheetah.",
        "To make a budget, list your income and expenses, and track your spending.",
        "Sleep is essential for memory, health, and overall well-being.",
        "You can create a website using HTML, CSS, and JavaScript, or with website builders.",
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Train for a marathon by gradually increasing your running distance and following a training plan.",
        "The Pythagorean theorem states that a² + b² = c² in a right triangle.",
        "Invest in stocks by opening a brokerage account and researching companies.",
        "The greenhouse effect is the warming of Earth's surface due to trapped heat from the atmosphere.",
        "Use calendars, to-do lists, and prioritization to organize your time better.",
        "Mount Everest is the tallest mountain in the world.",
        "Boil water, add pasta, cook until al dente, then drain.",
        "The meaning of life is a philosophical question with many interpretations.",
        "Reduce stress by exercising, meditating, and managing your time effectively.",
        "The Fibonacci sequence is a series where each number is the sum of the two preceding ones.",
        "Set up a Wi-Fi network by connecting a router to your modem and configuring it.",
        "RAM is temporary memory; ROM is permanent storage in computers.",
        "A cover letter should introduce yourself and explain why you're a good fit for the job.",
        "The heart pumps blood throughout the body, supplying oxygen and nutrients.",
        "Learn to draw by practicing basic shapes and studying drawing techniques.",
        "Supply and demand is an economic model of price determination in a market.",
        "Turn off your laptop, use compressed air, and gently clean the keyboard.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "Meditate by sitting quietly, focusing on your breath, and letting thoughts pass.",
        "Use repetition, visualization, and association to memorize information.",
        "Plant a tree by digging a hole, placing the sapling, and covering the roots with soil.",
        "Viruses require a host to reproduce; bacteria can live independently.",
        "To make coffee, brew ground coffee beans with hot water.",
        "Government provides order, security, and public services.",
        "To tie a tie, follow the steps for a simple knot like the four-in-hand.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Recycle by sorting materials and following local recycling guidelines.",
        "Photosynthesis is the process by which plants convert sunlight into energy."
    ]

    # Add 30 stock-related questions and answers
    stock_questions = [
        "What are stocks?",
        "How do I buy stocks?",
        "What is a stock exchange?",
        "How does the stock market work?",
        "What is a stockbroker?",
        "What is an IPO?",
        "How do dividends work?",
        "What is a stock split?",
        "What is a blue-chip stock?",
        "What is a penny stock?",
        "How do I analyze a stock?",
        "What is technical analysis?",
        "What is fundamental analysis?",
        "What is a stock portfolio?",
        "How do I diversify my investments?",
        "What is a bull market?",
        "What is a bear market?",
        "What is market capitalization?",
        "What is a P/E ratio?",
        "What is a limit order?",
        "What is a market order?",
        "What is short selling?",
        "What is a stop-loss order?",
        "What is a mutual fund?",
        "What is an ETF?",
        "How do I start investing in stocks?",
        "What are the risks of investing in stocks?",
        "What is portfolio rebalancing?",
        "What is insider trading?",
        "What is a stock index?"
    ]
    stock_answers = [
        "Stocks are shares of ownership in a company that can be bought and sold on the stock market.",
        "You can buy stocks through a brokerage account by placing buy orders for the stocks you want.",
        "A stock exchange is a marketplace where stocks are bought and sold, such as the NYSE or NASDAQ.",
        "The stock market is a system where investors buy and sell shares of publicly traded companies.",
        "A stockbroker is a professional or platform that facilitates the buying and selling of stocks for investors.",
        "An IPO, or Initial Public Offering, is when a company first sells its shares to the public.",
        "Dividends are payments made by a company to its shareholders, usually from profits.",
        "A stock split is when a company increases its number of shares, reducing the price per share proportionally.",
        "A blue-chip stock is a share in a large, reputable, and financially sound company.",
        "A penny stock is a stock that trades at a low price, typically outside of major market exchanges.",
        "To analyze a stock, review its financial statements, performance, and market trends.",
        "Technical analysis involves evaluating stocks based on price charts and trading volumes.",
        "Fundamental analysis involves assessing a company's financial health and business prospects.",
        "A stock portfolio is a collection of stocks owned by an investor.",
        "Diversifying investments means spreading your money across different assets to reduce risk.",
        "A bull market is a period when stock prices are rising or expected to rise.",
        "A bear market is a period when stock prices are falling or expected to fall.",
        "Market capitalization is the total value of a company's outstanding shares of stock.",
        "The P/E ratio compares a company's share price to its earnings per share.",
        "A limit order is an order to buy or sell a stock at a specific price or better.",
        "A market order is an order to buy or sell a stock immediately at the best available price.",
        "Short selling is selling borrowed stock with the intention of buying it back at a lower price.",
        "A stop-loss order automatically sells a stock when it reaches a certain price to limit losses.",
        "A mutual fund is an investment vehicle that pools money to buy a diversified portfolio of stocks and bonds.",
        "An ETF, or Exchange-Traded Fund, is a fund that trades on stock exchanges like a stock.",
        "To start investing in stocks, open a brokerage account and research companies to invest in.",
        "Risks of investing in stocks include market volatility, company performance, and economic factors.",
        "Portfolio rebalancing is adjusting your investments to maintain your desired asset allocation.",
        "Insider trading is the illegal practice of trading stocks based on non-public information.",
        "A stock index measures the performance of a group of stocks, like the S&P 500."
    ]

    sample_questions.extend(stock_questions)
    sample_answers.extend(stock_answers)

    conversations = []
    for i in range(80):
        user_input = sample_questions[i % len(sample_questions)]
        assistant_response = sample_answers[i % len(sample_answers)]
        bloated_metadata = {
            "session_id": f"sess_{10000+i}",
            "user_agent": "Mozilla/5.0 (TestAgent)",
            "ip_address": f"192.168.1.{100+i}",
            "timestamp_created": f"2024-01-15T10:{i:02d}:00Z",
            "timestamp_modified": f"2024-01-15T10:{i:02d}:05Z",
            "version": "1.0.0",
            "debug_info": {"line_number": i, "function": "process_request", "stack_trace": "..."},
            "performance_metrics": {"response_time_ms": 100+i, "memory_usage_mb": 20.0+i},
            "analytics_data": {"page_view": True, "user_segment": "test", "conversion_rate": 0.1+i/100},
            "system_info": {"os": "TestOS", "browser": "TestBrowser", "screen_resolution": "1024x768"},
            "extraneous_field_1": f"unnecessary_data_{i*3+1}",
            "extraneous_field_2": f"unnecessary_data_{i*3+2}",
            "extraneous_field_3": f"unnecessary_data_{i*3+3}",
            "unnecessary_metadata_1": "This field is completely irrelevant to the conversation",
            "unnecessary_metadata_2": "Another irrelevant field that bloats the context",
            "unnecessary_metadata_3": "Yet another field that should be filtered out",
            "debug_logs": f"DEBUG: Processing request at line {i}\nDEBUG: Memory allocation successful\nDEBUG: Response generated in {100+i}ms",
            "error_logs": "No errors in this conversation",
            "audit_trail": f"User authenticated at 10:{i:02d}:00\nRequest processed at 10:{i:02d}:05\nResponse sent at 10:{i:02d}:08",
            "cache_info": f"Cache hit rate: {80+i%20}%\nCache size: 256MB\nCache entries: {1000+i}",
            "load_balancer_info": f"Request routed to server-{i%5}\nLoad: {40+i%60}%\nResponse time: {100+i}ms"
        }
        conversations.append([(user_input, assistant_response, bloated_metadata)])
    
    print(f"📝 Populating Redis with {len(conversations)} bloated conversation threads...")
    
    # Generate timestamps over the past 30 days
    base_time = datetime.now() - timedelta(days=30)
    
    for thread_idx, conversation in enumerate(conversations):
        thread_id = f"conversation_{thread_idx + 1}"
        
        for msg_idx, (user_input, assistant_response, bloated_metadata) in enumerate(conversation):
            # Generate timestamp (spread over past 30 days)
            timestamp = base_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            message_id = f"{thread_id}:{timestamp.isoformat()}"
            
            # Prepare message data as a dict
            message_data = {
                'id': message_id,
                'timestamp': timestamp.isoformat(),
                'thread_id': thread_id,
                'user_input': user_input,
                'assistant_response': assistant_response,
                'tools_used': [],
                'metadata': bloated_metadata,
                'session_id': bloated_metadata.get('session_id', ''),
                'user_agent': bloated_metadata.get('user_agent', ''),
                'ip_address': bloated_metadata.get('ip_address', ''),
                'timestamp_created': bloated_metadata.get('timestamp_created', ''),
                'timestamp_modified': bloated_metadata.get('timestamp_modified', ''),
                'version': bloated_metadata.get('version', ''),
                'debug_info': bloated_metadata.get('debug_info', {}),
                'performance_metrics': bloated_metadata.get('performance_metrics', {}),
                'analytics_data': bloated_metadata.get('analytics_data', {}),
                'system_info': bloated_metadata.get('system_info', {}),
                'extraneous_field_1': bloated_metadata.get('extraneous_field_1', ''),
                'extraneous_field_2': bloated_metadata.get('extraneous_field_2', ''),
                'extraneous_field_3': bloated_metadata.get('extraneous_field_3', ''),
                'unnecessary_metadata_1': "This field is completely irrelevant to the conversation",
                'unnecessary_metadata_2': "Another irrelevant field that bloats the context",
                'unnecessary_metadata_3': "Yet another field that should be filtered out",
                'debug_logs': "DEBUG: Processing request at line 42\nDEBUG: Memory allocation successful\nDEBUG: Response generated in 150ms",
                'error_logs': "No errors in this conversation",
                'audit_trail': "User authenticated at 10:30:00\nRequest processed at 10:30:05\nResponse sent at 10:30:08",
                'cache_info': "Cache hit rate: 85%\nCache size: 256MB\nCache entries: 1024",
                'load_balancer_info': "Request routed to server-3\nLoad: 45%\nResponse time: 150ms"
            }
            
            # Store as HASH with full content and embedding_bin
            content_str = json.dumps(message_data, ensure_ascii=False, sort_keys=True)
            embedding = st_model.encode(content_str).tolist()
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            try:
                redis_client_bin.hset(f"message:{message_id}", mapping={
                    'content': content_str,
                    'embedding_bin': embedding_bytes
                })
                print(f"[OK] HSET: message:{message_id}")
                # Immediately verify
                stored = redis_client_bin.hgetall(f"message:{message_id}")
                print(f"[VERIFY] HGETALL: message:{message_id} => {stored}")
            except Exception as e:
                print(f"[ERR] HSET failed: message:{message_id} - {e}")
            try:
                redis_client_bin.zadd(f"thread:{thread_id}:messages", {message_id: timestamp.timestamp()})
                print(f"[OK] ZADD: thread:{thread_id}:messages -> {message_id}")
            except Exception as e:
                print(f"[ERR] ZADD failed: thread:{thread_id}:messages -> {message_id} - {e}")
    
    # Verify the data was stored
    total_messages = len(redis_client.keys("message:*"))
    total_embeddings = len(redis_client.keys("embedding:*"))
    total_threads = len(redis_client.keys("thread:*"))
    
    print(f"✅ Successfully populated Redis with BLOATED data:")
    print(f"   - {total_messages} messages with extraneous metadata")
    print(f"   - {total_embeddings} embeddings")
    print(f"   - {total_threads} conversation threads")
    print(f"   - {len(conversations)} different topics covered")
    
    print("\n🎯 Topics covered include:")
    topics = [
        "Technology & Programming (with bloated metadata)",
        "Mathematics & Calculations (with bloated metadata)", 
        "Health & Wellness (with bloated metadata)",
        "Business & Career (with bloated metadata)"
    ]
    
    for i, topic in enumerate(topics, 1):
        print(f"   {i}. {topic}")
    
    print("\n🚀 You can now test the LLM field selection with bloated data!")
    print("Run 'python advanced_memory_system.py' to see the LLM filter out irrelevant fields.")

    # --- DIAGNOSTICS: Print Redis connection info ---
    print(f"[DIAG] redis_client: host={redis_client.connection_pool.connection_kwargs.get('host')}, port={redis_client.connection_pool.connection_kwargs.get('port')}, db={redis_client.connection_pool.connection_kwargs.get('db')}")
    print(f"[DIAG] redis_client_bin: host={redis_client_bin.connection_pool.connection_kwargs.get('host')}, port={redis_client_bin.connection_pool.connection_kwargs.get('port')}, db={redis_client_bin.connection_pool.connection_kwargs.get('db')}")
    # After all writes, print total number of keys
    total_keys = redis_client.keys('*')
    print(f"[DIAG] Total keys in redis_client after population: {len(total_keys)}")
    total_keys_bin = redis_client_bin.keys('*')
    print(f"[DIAG] Total keys in redis_client_bin after population: {len(total_keys_bin)}")

if __name__ == "__main__":
    populate_bloated_memory() 