import os
import requests
import logging
import json
import time
import re
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import uuid
import numpy as np
from collections import deque
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import threading
from typing import Dict, List, Tuple, Optional
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load .env file
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Configure secret key
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is missing from the environment")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is missing from the environment")

# Allow CORS for all origins (for development purposes)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# In-memory storage for conversations (use PostgreSQL in production)
conversations = {}

# Initialize database connection
db_conn = None

# Get API Key from environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Missing API Key! Set it in the .env file.")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
API_HEADERS = {"x-goog-api-key": API_KEY}
# Initialize PostgreSQL connection with retry logic
def init_postgresql():
    """Initialize PostgreSQL connection with database creation if needed"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # First try to connect to the specific database
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", 5432),
                database=os.getenv("DB_NAME", "emotional_chat_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                sslmode=os.getenv("DB_SSLMODE", "disable"),
                connect_timeout=5
            )
            
            # Create tables if they don't exist
            with conn.cursor() as cur:
                # Users table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        user_name TEXT,
                        user_gender TEXT,
                        user_details JSONB,
                        emotional_state JSONB,
                        relationship_depth FLOAT,
                        conversation_history JSONB,
                        inside_jokes JSONB,
                        shared_experiences JSONB,
                        messages JSONB,
                        user_interests JSONB,
                        current_topic TEXT,
                        interest_scores JSONB,
                        ad_targets JSONB,
                        ad_profile JSONB,
                        ad_targets_updated_at TIMESTAMP,
                        last_active TIMESTAMP,
                        last_updated TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Conversations table for the memory system
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT REFERENCES users(user_id),
                        user_input TEXT,
                        ai_response TEXT,
                        emotions JSONB,
                        intensity FLOAT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User memory table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_memory (
                        user_id TEXT PRIMARY KEY REFERENCES users(user_id),
                        conversations JSONB,
                        interests JSONB,
                        personal_details JSONB,
                        relationship_level INTEGER,
                        interest_scores JSONB,
                        ad_targets JSONB,
                        ad_profile JSONB,
                        ad_targets_updated_at TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User reminders table for medicine reminders
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_reminders (
                        user_id TEXT PRIMARY KEY,
                        reminders JSONB,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Medication logs table for tracking adherence
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS medication_logs (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        reminder_id BIGINT,
                        taken BOOLEAN,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
            
            print("PostgreSQL initialized successfully")
            return conn
            
        except psycopg2.OperationalError as e:
            if "database" in str(e) and "does not exist" in str(e):
                print(f"Database does not exist, attempting to create it (attempt {attempt + 1})")
                create_database()
                time.sleep(retry_delay)
            else:
                print(f"PostgreSQL connection error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Running without database support.")
                    return None
        except Exception as e:
            print(f"Error initializing PostgreSQL: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Running without database support.")
                return None
    
    return None

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (without specifying a database)
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", 5432),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            sslmode=os.getenv("DB_SSLMODE", "disable")
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            db_name = os.getenv("DB_NAME", "emotional_chat_db")
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cur.fetchone()
            
            if not exists:
                # Create the database
                cur.execute(f"CREATE DATABASE {db_name}")
                print(f"Database {db_name} created successfully")
            else:
                print(f"Database {db_name} already exists")
                
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        if conn:
            conn.close()

# Medical Reminder Endpoints
@app.route('/api/reminders', methods=['GET', 'POST'])
def handle_reminders():
    """Handle medicine reminder operations"""
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id', 'default')
            # Get reminders from database or return empty list
            if db_conn:
                with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT reminders FROM user_reminders WHERE user_id = %s",
                        (user_id,)
                    )
                    result = cur.fetchone()
                    reminders = result['reminders'] if result else []
            else:
                reminders = []
            
            return jsonify({
                'success': True,
                'reminders': reminders
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            user_id = data.get('user_id', 'default')
            reminders = data.get('reminders', [])
            
            # Save reminders to database
            if db_conn:
                with db_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_reminders (user_id, reminders, last_updated)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id)
                        DO UPDATE SET reminders = %s, last_updated = %s
                        """,
                        (user_id, json.dumps(reminders), datetime.now(),
                         json.dumps(reminders), datetime.now())
                    )
                    db_conn.commit()
            
            return jsonify({
                'success': True,
                'message': 'Reminders saved successfully'
            })
            
    except Exception as e:
        logging.error(f"Error in reminder endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reminder/taken', methods=['POST'])
def mark_reminder_taken():
    """Mark a reminder as taken and update statistics"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        reminder_id = data.get('reminder_id')
        taken = data.get('taken', True)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Log the medication adherence
        if db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO medication_logs (user_id, reminder_id, taken, timestamp)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, reminder_id, taken, timestamp)
                )
                db_conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Reminder status logged successfully'
        })
        
    except Exception as e:
        logging.error(f"Error logging reminder status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





# Enhanced Interest categories for detection
INTEREST_CATEGORIES = {
    'sports': [
        'sports', 'athlete', 'game', 'fitness', 'exercise', 'gym', 'training', 'workout', 'nutrition', 'strength',
        'cardio', 'yoga', 'pilates', 'medal', 'championship', 'tournament', 'league', 'coach', 'team', 'victory',
        'score', 'goal', 'touchdown', 'home run', 'arena', 'stadium', 'olympics', 'marathon', 'sprint', 'weightlifting',
        'cycling', 'swimming', 'diving', 'gymnastics', 'tennis', 'volleyball', 'baseball', 'hockey', 'cricket', 'rugby',
        'golf', 'boxing', 'mma', 'wrestling', 'skiing', 'snowboarding', 'surfing', 'skateboarding', 'motorsports', 'esports'
    ],
    'food': [
        'food', 'cooking', 'recipe', 'cuisine', 'restaurant', 'chef', 'baking', 'eating', 'dish', 'meal', 'taste',
        'flavor', 'ingredient', 'menu', 'dining', 'gourmet', 'street food', 'fine dining', 'breakfast', 'lunch', 'dinner',
        'brunch', 'appetizer', 'entree', 'dessert', 'pastry', 'chocolate', 'coffee', 'tea', 'wine', 'beer', 'cocktail',
        'bar', 'cafe', 'bakery', 'butcher', 'farm to table', 'organic', 'vegan', 'vegetarian', 'gluten free', 'paleo',
        'keto', 'bbq', 'grill', 'saute', 'roast', 'simmer', 'garnish', 'plate', 'foodie', 'delicious', 'yummy'
    ],
    'travel': [
        'travel', 'tour', 'vacation', 'destination', 'adventure', 'explore', 'trip', 'journey', 'flight', 'hotel',
        'beach', 'resort', 'airbnb', 'hostel', 'sightseeing', 'landmark', 'culture', 'backpacking', 'souvenir', 'itinerary',
        'passport', 'visa', 'backpack', 'suitcase', 'airport', 'cruise', 'road trip', 'hiking', 'camping', 'safari',
        'wanderlust', 'getaway', 'island', 'mountain', 'city break', 'countryside', 'local', 'tourist', 'guidebook', 'review',
        'booking', 'departure', 'arrival', 'immigration', 'sunset', 'photography', 'memory', 'experience'
    ],
    'music': [
        'music', 'song', 'artist', 'band', 'concert', 'listen', 'album', 'playlist', 'genre', 'lyrics', 'melody',
        'harmony', 'beat', 'rhythm', 'tempo', 'bass', 'guitar', 'piano', 'drums', 'violin', 'saxophone', 'trumpet',
        'singer', 'vocal', 'rapper', 'dj', 'producer', 'soundtrack', 'vinyl', 'streaming', 'spotify', 'apple music',
        'headphones', 'speakers', 'festival', 'gig', 'tour', 'billboard', 'charts', 'rock', 'pop', 'jazz', 'classical',
        'hip hop', 'rap', 'electronic', 'edm', 'house', 'techno', 'indie', 'folk', 'r&b', 'soul', 'reggae'
    ],
    'movies': [
        'movie', 'film', 'cinema', 'actor', 'actress', 'director', 'hollywood', 'netflix', 'watch', 'scene', 'story',
        'plot', 'character', 'dialogue', 'script', 'screenplay', 'oscar', 'award', 'nomination', 'premiere', 'trailer',
        'teaser', 'review', 'critic', 'rating', 'genre', 'comedy', 'drama', 'action', 'thriller', 'horror', 'sci-fi',
        'fantasy', 'documentary', 'animation', 'blockbuster', 'indie film', 'sequel', 'prequel', 'franchise', 'marvel',
        'dc', 'star wars', 'director\'s cut', 'cinematography', 'special effects', 'vfx', 'stunt', 'costume', 'set design'
    ],
    'technology': [
        'tech', 'device', 'gadget', 'smartphone', 'computer', 'laptop', 'desktop', 'tablet', 'digital', 'innovation',
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'data science', 'big data', 'algorithm', 'cloud',
        'cloud computing', 'software', 'hardware', 'app', 'application', 'website', 'internet', 'web', 'cybersecurity',
        'hacking', 'privacy', 'crypto', 'cryptocurrency', 'blockchain', 'nft', 'metaverse', 'vr', 'virtual reality',
        'ar', 'augmented reality', 'robot', 'robotics', 'automation', 'iot', 'internet of things', 'startup', 'tech news',
        'review', 'tutorial', 'code', 'programming', 'developer', 'engineer', 'apple', 'google', 'microsoft', 'tesla'
    ],
    'gaming': [
        'game', 'gaming', 'video game', 'console', 'player', 'esports', 'stream', 'gamer', 'playstation', 'xbox', 'nintendo',
        'pc gaming', 'steam', 'epic games', 'controller', 'headset', 'vr gaming', 'mmorpg', 'rpg', 'fps', 'shooter',
        'battle royale', 'strategy', 'puzzle', 'indie game', 'title', 'release date', 'pre-order', 'dlc', 'expansion',
        'patch', 'update', 'graphics', 'gameplay', 'quest', 'level', 'boss', 'character', 'skin', 'mod', 'modding',
        'twitch', 'youtube gaming', 'streamer', 'pro gamer', 'tournament', 'competitive', 'leaderboard', 'score', 'save point'
    ],
    'reading': [
        'book', 'reading', 'novel', 'author', 'writer', 'literature', 'story', 'page', 'library', 'read', 'fiction',
        'non-fiction', 'biography', 'memoir', 'autobiography', 'sci-fi', 'fantasy', 'mystery', 'thriller', 'romance',
        'historical fiction', 'poetry', 'poem', 'essay', 'publisher', 'publishing', 'bookstore', 'bestseller', 'classic',
        'chapter', 'prose', 'plot', 'character', 'setting', 'theme', 'ebook', 'audiobook', 'kindle', 'book club',
        'review', 'bookstagram', 'bibliophile', 'manuscript', 'proofread', 'edit', 'genre', 'young adult', 'ya', 'children\'s'
    ],
    'art': [
        'art', 'painting', 'drawing', 'sketch', 'creative', 'design', 'museum', 'gallery', 'exhibition', 'artist',
        'sculpture', 'sculpting', 'portrait', 'landscape', 'abstract', 'modern art', 'contemporary art', 'classical art',
        'renaissance', 'impressionism', 'canvas', 'easel', 'paint', 'acrylic', 'watercolor', 'oil paint', 'pastel',
        'charcoal', 'pencil', 'illustration', 'digital art', 'print', 'pottery', 'ceramics', 'curator', 'art history',
        'masterpiece', 'folio', 'installment', 'mural', 'graffiti', 'street art', 'commission', 'artwork', 'aesthetic'
    ],
    'photography': [
        'photo', 'photography', 'camera', 'picture', 'shot', 'lens', 'digital', 'portrait', 'landscape', 'wildlife',
        'nature', 'street photography', 'wedding photography', 'dslr', 'mirrorless', 'shutter', 'aperture', 'iso',
        'exposure', 'composition', 'lighting', 'flash', 'tripod', 'filter', 'edit', 'photoshop', 'lightroom', 'preset',
        'album', 'print', 'frame', 'photographer', 'model', 'pose', 'session', 'bokeh', 'macro', 'zoom', 'wide angle',
        'vintage', 'analog', 'film camera', 'develop', 'darkroom', 'drone photography', 'astrophotography'
    ],
    'fashion': [
        'fashion', 'style', 'clothing', 'outfit', 'apparel', 'wear', 'trend', 'vogue', 'couture', 'designer',
        'model', 'runway', 'catwalk', 'collection', 'lookbook', 'street style', 'haute couture', 'accessory',
        'jewelry', 'shoes', 'heels', 'sneakers', 'boots', 'handbag', 'purse', 'sunglasses', 'watch', 'hat',
        'dress', 'skirt', 'blouse', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'suit', 'tailoring', 'fabric',
        'silk', 'cotton', 'denim', 'leather', 'wardrobe', 'stylist', 'blogger', 'influencer', 'luxury', 'retail'
    ],
    'health_wellness': [
        'health', 'wellness', 'wellbeing', 'selfcare', 'meditation', 'mindfulness', 'yoga', 'mental health', 'therapy',
        'counseling', 'nutrition', 'diet', 'vitamin', 'supplement', 'organic', 'fitness', 'workout', 'sleep', 'hygiene',
        'detox', 'holistic', 'alternative medicine', 'acupuncture', 'chiropractor', 'massage', 'spa', 'relaxation',
        'stress relief', 'anxiety', 'depression', 'motivation', 'habits', 'routine', 'journaling', 'gratitude', 'balance',
        'immune system', 'superfood', 'smoothie', 'juice', 'wellness retreat', 'personal growth', 'mindset'
    ],
    'science':[
        'science', 'scientific', 'research', 'discovery', 'experiment', 'theory', 'hypothesis', 'scientist', 'biology',
        'chemistry', 'physics', 'astronomy', 'astrophysics', 'geology', 'meteorology', 'oceanography', 'psychology',
        'neuroscience', 'genetics', 'evolution', 'darwin', 'einstein', 'nasa', 'space', 'planet', 'star', 'galaxy',
        'microscope',
        'nobel prize', 'quantum', 'particle', 'element', 'molecule', 'atom', 'cell', 'dna', 'organism', 'ecosystem'
    ],
    'history': [
        'history', 'historical', 'past', 'era', 'period', 'empire', 'civilization', 'archeology', 'artifact', 'museum',
        'war', 'battle', 'king', 'queen', 'emperor', 'pharaoh', 'president', 'prime minister', 'revolution', 'independence',
        'medieval', 'renaissance', 'ancient', 'prehistoric', 'world war', 'wwii', 'wwi', 'middle ages', 'victorian',
        'colonial', 'dinosaurs', 'fossil', 'excavation', 'document', 'archive', 'biography', 'historian', 'timeline',
        'monument', 'landmark', 'castle', 'pyramid', 'temple', 'ruins', 'heritage', 'culture', 'tradition'
    ],
    'business_finance': [
        'business', 'finance', 'economy', 'economic', 'market', 'stock', 'investment', 'invest', 'trader', 'trading',
        'wall street', 'nyse', 'nasdaq', 'crypto', 'bitcoin', 'ethereum', 'blockchain', 'startup', 'entrepreneur',
        'entrepreneurship', 'venture capital', 'vc', 'angel investor', 'business plan', 'marketing', 'sales', 'profit',
        'revenue', 'loss', 'growth', 'strategy', 'management', 'ceo', 'cfo', 'cto', 'company', 'corporation', 'inc',
        'llc', 'bank', 'banking', 'loan', 'interest rate', 'credit', 'debit', 'tax', 'accounting', 'audit', 'wealth',
        'financial freedom', 'retirement', '401k', 'portfolio'
    ],
    'education': [
        'education', 'educational', 'learn', 'learning', 'student', 'teacher', 'professor', 'school', 'university',
        'college', 'campus', 'degree', 'bachelor', 'master', 'phd', 'doctorate', 'course', 'class', 'lesson', 'lecture',
        'homework', 'assignment', 'exam', 'test', 'grade', 'gpa', 'scholarship', 'tuition', 'graduate', 'alumni',
        'textbook', 'study', 'research', 'academic', 'literacy', 'numeracy', 'stem', 'steam', 'homeschool', 'online learning',
        'mooc', 'edtech', 'principal', 'dean', 'faculty', 'major', 'minor', 'thesis', 'dissertation'
    ],
    'diy_crafts': [
        'diy', 'do it yourself', 'craft', 'crafting', 'project', 'build', 'make', 'create', 'handmade', 'homemade',
        'upcycle', 'repurpose', 'woodworking', 'carpentry', 'sewing', 'knitting', 'crochet', 'embroidery', 'quilting',
        'scrapbooking', 'pottery', 'ceramics', 'clay', 'painting', 'decoupage', 'macrame', 'candle making', 'soap making',
        'jewelry making', 'beads', 'tools', 'hammer', 'saw', 'drill', 'screwdriver', 'nail', 'screw', 'glue', 'tape',
        'measure', 'plan', 'design', 'instructions', 'tutorial', 'hobby', 'workshop', 'shed', 'makerspace'
    ],
    'home_garden': [
        'home', 'garden', 'gardening', 'house', 'interior', 'exterior', 'design', 'decor', 'decoration', 'furniture',
        'couch', 'sofa', 'table', 'chair', 'bed', 'mattress', 'cabinet', 'shelf', 'lighting', 'lamp', 'rug', 'curtain',
        'appliance', 'renovation', 'remodel', 'diy', 'landscaping', 'lawn', 'yard', 'patio', 'deck', 'plant', 'flower',
        'vegetable', 'herb', 'organic', 'soil', 'compost', 'pot', 'seed', 'grow', 'prune', 'water', 'weed', 'harvest',
        'home improvement', 'hardware store', 'real estate', 'property', 'mortgage', 'rent', 'lease', 'clean', 'organize'
    ],
    'pets': [
        'pet', 'dog', 'puppy', 'cat', 'kitten', 'bird', 'parrot', 'fish', 'aquarium', 'tank', 'hamster', 'rabbit',
        'guinea pig', 'reptile', 'lizard', 'snake', 'turtle', 'veterinarian', 'vet', 'animal', 'adopt', 'adoption',
        'breed', 'purebred', 'mutt', 'rescue', 'shelter', 'food', 'treat', 'toy', 'leash', 'collar', 'harness',
        'grooming', 'bath', 'walk', 'park', 'training', 'obedience', 'trick', 'pet health', 'vaccine', 'flea', 'tick',
        'pet friendly', 'pet owner', 'dog lover', 'cat person', 'pet care', 'animal welfare'
    ],
    'cars_motorcycles': [
        'car', 'auto', 'automobile', 'vehicle', 'motorcycle', 'bike', 'suv', 'truck', 'sedan', 'coupe', 'convertible',
        'luxury car', 'sports car', 'classic car', 'electric vehicle', 'ev', 'tesla', 'hybrid', 'engine', 'horsepower',
        'torque', 'mpg', 'fuel efficiency', 'transmission', 'manual', 'automatic', 'tire', 'wheel', 'rim', 'alloy',
        'performance', 'race', 'racing', 'driving', 'driver', 'road trip', 'off-road', '4x4', 'maintenance', 'repair',
        'modification', 'tuning', 'detailing', 'wash', 'wax', 'showroom', 'test drive', 'dealer', 'brand', 'model', 'year'
    ],
    'astrology_spirituality': [
        'astrology', 'zodiac', 'horoscope', 'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio',
        'sagittarius', 'capricorn', 'aquarius', 'pisces', 'birth chart', 'natal chart', 'moon sign', 'rising sign',
        'tarot', 'tarot card', 'reading', 'psychic', 'medium', 'spirituality', 'spiritual', 'energy', 'aura', 'chakra',
        'meditation', 'mindfulness', 'yoga', 'reiki', 'crystal', 'healing', 'manifestation', 'law of attraction',
        'universe', 'karma', 'dharma', 'nirvana', 'enlightenment', 'faith', 'belief', 'ritual', 'sage', 'palms'
    ],
    'outdoors_adventure': [
        'outdoors', 'adventure', 'wilderness', 'nature', 'hiking', 'camping', 'backpacking', 'climbing', 'rock climbing',
        'mountaineering', 'summit', 'trail', 'path', 'national park', 'forest', 'wood', 'jungle', 'desert', 'canyon',
        'river', 'lake', 'ocean', 'kayaking', 'canoeing', 'paddling', 'rafting', 'sailing', 'scuba', 'diving', 'snorkeling',
        'surfing', 'windsurfing', 'kitesurfing', 'skiing', 'snowboarding', 'sledding', 'ice climbing', 'exploration',
        'expedition', 'trek', 'outdoor gear', 'tent', 'sleeping bag', 'hiking boots', 'compass', 'map', 'gps', 'survival'
    ],
       'parenting_family': [
        'parenting', 'parent', 'mother', 'mom', 'father', 'dad', 'family', 'children', 'child', 'kids', 'baby',
        'toddler', 'newborn', 'pregnancy', 'pregnant', 'expecting', 'due date', 'birth', 'labor', 'delivery',
        'breastfeeding', 'formula', 'diaper', 'pacifier', 'onesie', 'childcare', 'daycare', 'nanny', 'stay at home',
        'parenting tips', 'discipline', 'tantrum', 'milestone', 'crawling', 'walking', 'talking', 'potty training',
        'school', 'ptsa', 'homework help', 'playdate', 'sibling', 'twins', 'multiples', 'family vacation', 'quality time',
        'co-parenting', 'foster care', 'adoption', 'grandparent', 'aunt', 'uncle', 'cousin', 'family game night'
    ],
    'comics_anime': [
        'comic', 'comics', 'graphic novel', 'manga', 'anime', 'superhero', 'marvel', 'dc comics', 'batman', 'superman',
        'spiderman', 'ironman', 'captain america', 'avengers', 'x-men', 'watchmen', 'villain', 'hero', 'issue', 'volume',
        'panel', 'artwork', 'cover art', 'variant cover', 'collector', 'box', 'subscription', 'writer', 'artist', 'inker',
        'letterer', 'cosplay', 'convention', 'comic-con', 'otaku', 'weeb', 'shonen', 'seinen', 'shojo', 'isekai', 'mecha',
        'studio ghibli', 'nakama', 'kawaii', 'senpai', 'tsundere', 'arc', 'filler', 'dub', 'sub', 'amv', 'fan service'
    ],
    'dance_performance': [
        'dance', 'dancing', 'ballet', 'contemporary', 'jazz', 'hip hop', 'breakdancing', 'breaking', 'b-boy', 'b-girl',
        'tap', 'salsa', 'tango', 'ballroom', 'swing', 'latin', 'choreography', 'choreographer', 'dancer', 'performance',
        'recital', 'show', 'routine', 'rehearsal', 'practice', 'technique', 'pirouette', 'plié', 'flexibility', 'grace',
        'rhythm', 'music video', 'flash mob', 'dance crew', 'competition', 'battle', 'judge', 'costume', 'leotard',
        'pointe shoes', 'dance studio', 'broadway', 'musical theater', 'acting', 'theatre', 'play', 'improv', 'improvisation'
    ],
    'philosophy_ethics': [
        'philosophy', 'philosophical', 'ethics', 'ethical', 'moral', 'morality', 'existentialism', 'stoicism', 'nihilism',
        'absurdism', 'logic', 'reason', 'critical thinking', 'argument', 'debate', 'discourse', 'theory', 'thinker',
        'plato', 'aristotle', 'socrates', 'nietzsche', 'kant', 'descartes', 'camus', 'sartre', 'confucius', 'lao tzu',
        'meaning of life', 'free will', 'determinism', 'consciousness', 'reality', 'perception', 'truth', 'knowledge',
        'epistemology', 'metaphysics', 'ontology', 'aesthetics', 'political philosophy', 'justice', 'rights', 'duty',
        'virtue', 'good and evil', 'the self', 'identity', 'phenomenology', 'deconstruction'
    ],
    'podcasts_audio': [
        'podcast', 'podcasts', 'audio', 'episode', 'show', 'host', 'interview', 'conversation', 'narrative', 'storytelling',
        'true crime', 'comedy', 'news', 'politics', 'business', 'educational', 'interview', 'roundtable', 'discussion',
        'serial', 'season', 'subscribe', 'download', 'stream', 'listener', 'review', 'rating', 'microphone', 'recording',
        'editing', 'producer', 'rss feed', 'spotify', 'apple podcasts', 'google podcasts', 'audible', 'audiobook',
        'narration', 'voice actor', 'sound design', 'earphones', 'headphones', 'commute', 'background noise', 'patreon'
    ],
    'sustainability_environment': [
        'sustainability', 'sustainable', 'environment', 'environmental', 'eco', 'green', 'climate', 'climate change',
        'global warming', 'planet', 'earth', 'conservation', 'preservation', 'protect', 'renewable', 'solar', 'wind',
        'hydro', 'energy', 'carbon', 'carbon footprint', 'emissions', 'pollution', 'plastic', 'waste', 'recycle',
        'recycling', 'compost', 'composting', 'zero waste', 'minimalism', 'ethical', 'fair trade', 'local', 'organic',
        'biodiversity', 'deforestation', 'ocean', 'cleanup', 'activism', 'activist', 'greta thunberg', 'policy', 'un sdgs'
    ],
    'volunteering_activism': [
        'volunteer', 'volunteering', 'activism', 'activist', 'charity', 'nonprofit', 'ngo', 'donate', 'donation',
        'fundraiser', 'fundraising', 'cause', 'social justice', 'human rights', 'civil rights', 'equality', 'diversity',
        'inclusion', 'community', 'serve', 'service', 'help', 'support', 'advocate', 'advocacy', 'protest', 'march',
        'rally', 'petition', 'sign', 'awareness', 'campaign', 'movement', 'ngo', 'non profit', 'philanthropy', 'philanthropist',
        'give back', 'make a difference', 'social change', 'grassroots', 'local community', 'soup kitchen', 'homelessness',
        'animal shelter', 'cleanup', 'mentor', 'mentorship'
    ],
    'frugality_deals': [
        'frugal', 'frugality', 'deal', 'deals', 'discount', 'coupon', 'sale', 'clearance', 'save', 'saving', 'money',
        'budget', 'budgeting', 'finance', 'financial', 'cheap', 'inexpensive', 'affordable', 'value', 'thrifty', 'thrift',
        'thrift store', 'secondhand', 'used', 'vintage', 'refurbished', 'free', 'rebate', 'cashback', 'reward', 'points',
        'hack', 'tip', 'trick', 'maximize', 'minimize cost', 'do it yourself', 'diy', 'generic', 'store brand', 'compare prices',
        'price match', 'black friday', 'cyber monday', 'prime day', 'wholesale', 'bulk', 'couponing', 'extreme couponing'
    ],
    'luxury_lifestyle': [
        'luxury', 'luxurious', 'premium', 'high-end', 'exclusive', 'couture', 'designer', 'bespoke', 'custom', 'opulent',
        'lavish', 'extravagant', 'wealth', 'rich', 'affluent', 'millionaire', 'billionaire', 'private', 'jet', 'yacht',
        'limousine', 'chauffeur', 'mansion', 'estate', 'penthouse', 'villa', 'chateau', 'concierge', 'butler', 'resort',
        'spa', 'fine dining', 'gourmet', 'champagne', 'caviar', 'truffle', 'haute couture', 'watch', 'rolex', 'patek',
        'diamond', 'jewelry', 'art collection', 'auction', 'sotheby\'s', 'christie\'s', 'members only', 'vip', 'status'
    ],
    'career_development': [
        'career', 'job', 'profession', 'work', 'office', 'corporate', 'business', 'professional', 'development',
        'skill', 'skills', 'training', 'workshop', 'seminar', 'webinar', 'networking', 'connect', 'linkedin',
        'resume', 'cv', 'cover letter', 'interview', 'hire', 'hiring', 'recruiter', 'recruitment', 'application',
        'candidate', 'employee', 'employer', 'promotion', 'raise', 'salary', 'compensation', 'benefits', 'leadership',
        'management', 'manager', 'team', 'collaboration', 'productivity', 'efficiency', 'remote work', 'work from home',
        'freelance', 'contract', 'gig economy', 'side hustle', 'entrepreneur', 'mentor', 'mentorship', 'career change'
    ],
    'humor_memes': [
        'humor', 'humorous', 'funny', 'comedy', 'joke', 'meme', 'memes', 'lol', 'laugh', 'laughing', 'hilarious',
        'satire', 'parody', 'sarcasm', 'wit', 'witty', 'pun', 'dad joke', 'stand-up', 'comedian', 'improv', 'sketch',
        'viral', 'trending', 'internet culture', 'reaction', 'reaction video', 'fail', 'fail video', 'prank', 'troll',
        'trolling', 'wholesome', 'absurd', 'random', 'niche meme', 'dank', 'template', 'caption', 'gif', 'giphy',
        'reddit', '9gag', 'ifunny', 'twitter', 'tiktok', 'vine', 'shitpost', 'shitposting', 'lighthearted', 'feel good'
    ],
    'politics_current_events': [
        'politics', 'political', 'government', 'policy', 'law', 'legislation', 'bill', 'act', 'congress', 'senate',
        'parliament', 'president', 'prime minister', 'election', 'vote', 'voting', 'campaign', 'candidate', 'democrat',
        'republican', 'liberal', 'conservative', 'left', 'right', 'centrist', 'activism', 'protest', 'rally', 'debate',
        'news', 'current events', 'headline', 'breaking news', 'world affairs', 'international relations', 'diplomacy',
        'economy', 'tax', 'healthcare', 'education', 'immigration', 'climate policy', 'supreme court', 'justice', 'rights',
        'freedom', 'democracy', 'geopolitics', 'sanctions', 'treaty', 'summit', 'sanctions', 'eurozone', 'brexit'
    ],
    'minimalism_simple_living': [
        'minimalism', 'minimalist', 'simple living', 'simplicity', 'essentialism', 'declutter', 'decluttering', 'organize',
        'organization', 'tidying', 'marie kondo', 'spark joy', 'konmari', 'downsize', 'downsizing', 'less is more',
        'intentional', 'intentional living', 'mindful', 'mindfulness', 'capsule wardrobe', 'quality over quantity',
        'possessions', 'own less', 'freedom', 'digital minimalism', 'slow living', 'quiet', 'peace', 'calm', 'serene',
        'aesthetic', 'neutral', 'clean lines', 'function over form', 'tiny house', 'van life', 'self sufficiency', 'frugal',
        'sustainable', 'consumption', 'anti consumerism', 'mindful spending', 'gratitude', 'contentment'
    ],
    'writing_poetry': [
        'writing', 'write', 'writer', 'author', 'poetry', 'poem', 'poet', 'prose', 'fiction', 'non-fiction', 'novel',
        'short story', 'essay', 'blog', 'journal', 'journaling', 'memoir', 'biography', 'screenplay', 'script', 'dialogue',
        'character', 'plot', 'setting', 'theme', 'genre', 'manuscript', 'draft', 'edit', 'editing', 'proofread', 'revise',
        'revision', 'publish', 'publishing', 'query', 'agent', 'publisher', 'self-publish', 'nanowrimo', 'writer\'s block',
        'inspiration', 'creative', 'metaphor', 'simile', 'imagery', 'rhyme', 'meter', 'haiku', 'sonnet', 'free verse',
        'spoken word', 'open mic', 'workshop', 'critique'
    ],
    'drones_tech': [
        'drone', 'drones', 'uav', 'quadcopter', 'fpv', 'first person view', 'racing drone', 'camera drone', 'aerial',
        'photography', 'videography', 'footage', 'pilot', 'remote', 'controller', 'gps', 'autonomous', 'flight',
        'battery', 'propeller', 'gimbal', '4k', 'hd', 'regulation', 'faa', 'part 107', 'commercial', 'hobbyist',
        'race', 'obstacle', 'trick', 'stunt', 'build', 'custom', 'frame', 'esc', 'flight controller', 'betaflight',
        'dji', 'autel', 'parrot', 'yuneec', 'range', 'signal', 'telemetry', 'mod', 'community', 'meetup'
    ],
    'board_games_tabletop': [
        'board game', 'board games', 'tabletop', 'game night', 'strategy', 'eurogame', 'ameritrash', 'card game',
        'deck', 'dice', 'miniature', 'minis', 'painting', 'terrain', 'dungeons and dragons', 'd&d', 'dnd', 'rpg',
        'pathfinder', 'warhammer', 'mtg', 'magic the gathering', 'pokemon', 'yu-gi-oh', 'chess', 'checkers', 'backgammon',
        'catan', 'ticket to ride', 'pandemic', 'arkham horror', 'gloomhaven', 'kickstarter', 'gamefound', 'rulebook',
        'expansion', 'kickstarter', 'meeple', 'player count', 'cooperative', 'competitive', 'solo', 'dm', 'dungeon master',
        'campaign', 'one-shot', 'adventure', 'character sheet', 'roll for initiative'
    ],
    'mixology_craft_beer': [
        'mixology', 'cocktail', 'bartender', 'bar', 'spirit', 'liquor', 'vodka', 'gin', 'rum', 'tequila', 'whiskey',
        'whisky', 'bourbon', 'scotch', 'vermouth', 'bitters', 'syrup', 'shake', 'stir', 'muddle', 'garnish', 'twist',
        'recipe', 'classic', 'innovation', 'craft', 'distillery', 'distill', 'aged', 'neat', 'on the rocks', 'shot',
        'beer', 'craft beer', 'ipa', 'ale', 'lager', 'stout', 'pilsner', 'sour', 'brewery', 'brewing', 'brewpub',
        'taproom', 'flight', 'hops', 'malt', 'yeast', 'fermentation', 'homebrew', 'keg', 'nitro', 'cask', 'abv', 'ibu'
    ],
    'home_automation_diy_tech': [
        'smart home', 'home automation', 'iot', 'internet of things', 'smart device', 'voice assistant', 'alexa',
        'google home', 'siri', 'homekit', 'smart speaker', 'smart display', 'smart light', 'philips hue', 'smart bulb',
        'smart plug', 'smart switch', 'smart thermostat', 'nest', 'ecobee', 'smart lock', 'smart doorbell', 'ring',
        'security camera', 'video doorbell', 'motion sensor', 'automation', 'routine', 'scene', 'hub', 'bridge',
        'zigbee', 'z-wave', 'wifi', 'bluetooth', 'diy', 'project', 'raspberry pi', 'arduino', 'homebridge', 'open source',
        'privacy', 'security', 'integration', 'compatibility', 'tech support', 'setup', 'configuration'
    ],
    'mycology_foraging': [
        'mycology', 'mushroom', 'mushrooms', 'fungi', 'fungus', 'foraging', 'forage', 'wild edible', 'hunting',
        'identification', 'species', 'chanterelle', 'morel', 'porcini', 'shiitake', 'oyster mushroom', 'psilocybin',
        'mycelium', 'spore', 'spore print', 'gills', 'cap', 'stem', 'toxic', 'poisonous', 'edible', 'medicinal',
        'forest', 'woods', 'guidebook', 'expert', 'caution', 'sustainability', 'leave no trace', 'cooking', 'frying',
        'drying', 'preserving', 'taxonomy', 'ecology', 'decomposition', 'symbiosis', 'lichens', 'truffle', 'truffle hunting'
    ],
    'urban_exploration': [
        'urban exploration', 'urbex', 'abandoned', 'derelict', 'asylum', 'hospital', 'factory', 'warehouse', 'school',
        'hotel', 'mall', 'ghost town', 'ruins', 'decay', 'rotting', 'graffiti', 'peeling paint', 'nature reclaiming',
        'exploration', 'adventure', 'photography', 'history', 'past', 'forgotten', 'access', 'rooftopping', 'roof access',
        'drainage', 'tunnels', 'catacombs', 'infiltration', 'safety', 'risk', 'caution', 'trespassing', 'legal',
        'preservation', 'documentation', 'aesthetic', 'liminal space', 'eerie', 'spooky', 'atmosphere', 'haunted'
    ],
    'lettering_calligraphy': [
        'calligraphy', 'lettering', 'hand lettering', 'typography', 'font', 'script', 'cursive', 'brush', 'brush pen',
        'ink', 'nib', 'dip pen', 'pointed pen', 'copperplate', 'spencerian', 'gothic', 'blackletter', 'modern calligraphy',
        'watercolor', 'flourish', 'ascender', 'descender', 'x-height', 'practice', 'drills', 'worksheet', 'alphabet',
        'quote', 'design', 'layout', 'composition', 'wedding invite', 'addressing', 'envelope', 'place card', 'logotype',
        'signage', 'handmade', 'artisan', 'skill', 'patience', 'relaxing', 'mindful', 'community', 'instagram'
    ],
    'scuba_diving_snorkeling': [
        'scuba', 'scuba diving', 'diving', 'snorkeling', 'snorkel', 'ocean', 'sea', 'reef', 'coral', 'coral reef',
        'tropical', 'fish', 'marine life', 'shark', 'turtle', 'ray', 'dolphin', 'nudibranch', 'wreck', 'shipwreck',
        'dive computer', 'bcd', 'regulator', 'tank', 'air', 'nitrox', 'wetsuit', 'drysuit', 'mask', 'fins', 'snorkel',
        'depth', 'pressure', 'equalize', 'buoyancy', 'padi', 'naui', 'ssi', 'certification', 'open water', 'advanced',
        'divemaster', 'instructor', 'safety', 'decompression', 'dive plan', 'buddy', 'boat', 'liveaboard', 'underwater photography'
    ],
    'vintage_antiques': [
        'vintage', 'antique', 'antiques', 'retro', 'collectible', 'collecting', 'thrifting', 'thrift store', 'fleamarket',
        'estate sale', 'auction', 'ebay', 'etsy', 'history', 'past', 'era', 'art deco', 'victorian', 'mid-century modern',
        'mcm', 'furniture', 'lighting', 'lamp', 'glassware', 'china', 'porcelain', 'silver', 'jewelry', 'watch', 'clock',
        'vinyl record', 'camera', 'typewriter', 'radio', 'art', 'poster', 'advertisement', 'sign', 'restoration', 'refinish',
        'repair', 'patina', 'authentic', 'rare', 'value', 'appraisal', 'decor', 'nostalgia', 'character', 'story'
    ],
    'permaculture_gardening': [
        'permaculture', 'gardening', 'garden', 'organic', 'sustainable', 'regenerative', 'food forest', 'companion planting',
        'soil health', 'compost', 'composting', 'vermicompost', 'worms', 'mulch', 'no-till', 'raised bed', 'hugelkultur',
        'rainwater', 'harvesting', 'irrigation', 'drip', 'seed', 'seed saving', 'heirloom', 'seedling', 'transplant',
        'harvest', 'yield', 'pollinator', 'bees', 'beneficial insects', 'pest', 'natural pest control', 'weed', 'chicken',
        'coop', 'homestead', 'self-sufficiency', 'climate', 'microclimate', 'zone', 'design', 'ethics', 'earth care', 'people care'
    ],
    'virtual_reality_ar': [
        'virtual reality', 'vr', 'augmented reality', 'ar', 'mixed reality', 'mr', 'metaverse', 'headset', 'oculus',
        'quest', 'rift', 'htc vive', 'valve index', 'playstation vr', 'controller', 'hand tracking', 'roomscale',
        'immersion', 'immersive', 'simulation', 'experience', 'game', 'gaming', 'app', 'application', 'social vr',
        'vrchat', 'rec room', 'education', 'training', 'architecture', 'design', '360 video', 'photosphere', 'development',
        'unity', 'unreal engine', '3d modeling', 'motion sickness', 'vr legs', 'fov', 'resolution', 'refresh rate', 'wireless'
    ],
    'kpop_jpop': [
        'kpop', 'k-pop', 'jpop', 'j-pop', 'korean pop', 'japanese pop', 'idol', 'group', 'band', 'boy group', 'girl group',
        'bts', 'blackpink', 'twice', 'exo', 'red velvet', 'stray kids', 'nct', 'seventeen', 'album', 'single', 'comeback',
        'mv', 'music video', 'dance practice', 'choreography', 'fandom', 'stan', 'bias', 'bias wrecker', 'lightstick',
        'concert', 'merch', 'merchandise', 'agency', 'sm entertainment', 'yg entertainment', 'jyp entertainment', 'hybe',
        'arakawa', 'akb48', 'perfume', 'babymetal', 'yoasobi', 'anime opening', 'ending', 'ost', 'fanchant', 'subunit'
    ],
    'tea_coffee_connoisseur': [
        'tea', 'coffee', 'connoisseur', 'specialty', 'artisan', 'brew', 'brewing', 'pour over', 'french press', 'aeropress',
        'chemex', 'espresso', 'machine', 'grinder', 'grind', 'beans', 'roast', 'roasting', 'light roast', 'dark roast',
        'single origin', 'blend', 'latte', 'cappuccino', 'macchiato', 'cold brew', 'nitro', 'cupping', 'tasting', 'notes',
        'aroma', 'acidity', 'body', 'finish', 'barista', 'cafe', 'tea leaves', 'loose leaf', 'green tea', 'black tea', 'oolong',
        'white tea', 'pu-erh', 'herbal tea', 'tisane', 'gaiwan', 'teapot', 'infuser', 'steep', 'temperature', 'ceremony'
    ],
    'aviation_spaceflight': [
        'aviation', 'airplane', 'aircraft', 'airline', 'airport', 'pilot', 'flying', 'flight', 'boeing', 'airbus',
        'cessna', 'general aviation', 'cockpit', 'atc', 'air traffic control', 'takeoff', 'landing', 'approach',
        'runway', 'taxi', 'turbulence', 'ceiling', 'visibility', 'flight simulator', 'x-plane', 'microsoft flight sim',
        'fs2020', 'space', 'spaceflight', 'nasa', 'spacex', 'rocket', 'launch', 'orbit', 'iss', 'international space station',
        'astronaut', 'cosmonaut', 'taikonaut', 'mission', 'capsule', 'starship', 'falcon', 'blue origin', 'virgin galactic',
        'elon musk', 'exploration', 'mars', 'moon', 'artemis', 'apollo', 'hubble', 'telescope', 'aerodynamics'
    ],
    'standup_comedy_improv': [
        'stand-up', 'standup', 'comedy', 'improv', 'improvisation', 'comedian', 'club', 'open mic', 'set', 'routine',
        'joke', 'punchline', 'setup', 'timing', 'delivery', 'crowd work', 'observational', 'satire', 'sketch', 'skit',
        'character', 'impression', 'voice', 'wit', 'roast', 'comedy central', 'netflix special', 'hour', 'special',
        'writer', 'writing', 'premise', 'tag', 'act out', 'mic', 'stage', 'light', 'heckler', 'bomb', 'kill', 'crush',
        'theater sports', 'whose line', 'second city', 'ucb', 'groundlings', 'yes and', 'scene', 'game', 'offer'
    ],
    'extreme_sports_adrenaline': [
        'extreme sports', 'adrenaline', 'thrill', 'action', 'danger', 'risk', 'adventure', 'bmx', 'moto', 'motocross',
        'freestyle', 'trick', 'stunt', 'jump', 'ramp', 'halfpipe', 'vert', 'big air', 'skateboard', 'skateboarding',
        'wakeboard', 'wakeboarding', 'wakeskate', 'snowboard', 'snowboarding', 'free solo', 'rock climbing', 'bouldering',
        'base jumping', 'parachute', 'paragliding', 'wingsuit', 'wingsuit flying', 'sky diving', 'skydive', 'bungee jump',
        'whitewater', 'kayaking', 'rafting', 'cave diving', 'exploration', 'endurance', 'challenge', 'fear', 'conquer',
        'competition', 'x games', 'red bull', 'athlete', 'daredevil'
    ],
    'mythology_folklore': [
        'mythology', 'myth', 'folklore', 'legend', 'tale', 'story', 'god', 'goddess', 'deity', 'pantheon', 'hero',
        'creature', 'monster', 'dragon', 'unicorn', 'griffin', 'greek', 'zeus', 'hera', 'apollo', 'athena', 'norse',
        'odin', 'thor', 'loki', 'valhalla', 'egyptian', 'ra', 'isis', 'osiris', 'anubis', 'celtic', 'druid', 'fairy',
        'asian', 'hindu', 'vishnu', 'shiva', 'kali', 'chinese', 'japanese', 'shinto', 'kami', 'urban legend', 'ghost story',
        'oral tradition', 'culture', 'belief', 'symbolism', 'archetype', 'campfire', 'storytelling', 'epic', 'saga'
    ],
    'cryptocurrency_defi': [
        'cryptocurrency', 'crypto', 'bitcoin', 'btc', 'ethereum', 'eth', 'altcoin', 'blockchain', 'distributed ledger',
        'mining', 'miner', 'proof of work', 'proof of stake', 'pos', 'defi', 'decentralized finance', 'exchange',
        'binance', 'coinbase', 'kraken', 'wallet', 'hot wallet', 'cold wallet', 'ledger', 'trezor', 'nft', 'non-fungible token',
        'token', 'airdrop', 'ico', 'ido', 'smart contract', 'solidity', 'gas', 'gas fee', 'transaction', 'hash', 'halving',
        'bull market', 'bear market', 'hODL', 'to the moon', 'whale', 'satoshis', 'regulation', 'sec', 'volatility', 'investment'
    ],
    'linguistics_languages': [
        'linguistics', 'language', 'languages', 'polyglot', 'grammar', 'syntax', 'semantics', 'phonetics', 'phonology',
        'morphology', 'etymology', 'origin', 'word', 'vocabulary', 'accent', 'dialect', 'slang', 'jargon', 'lingo',
        'translation', 'interpreter', 'bilingual', 'multilingual', 'learn', 'learning', 'duolingo', 'babbel', 'rosetta stone',
        'immersion', 'fluent', 'fluency', 'conjugate', 'decline', 'noun', 'verb', 'adjective', 'alphabet', 'script',
        'cyrillic', 'arabic', 'kanji', 'hieroglyph', 'conlang', 'constructed language', 'esperanto', 'klingon', 'high valyrian',
        'cognitive', 'communication', 'sociolinguistics', 'neurolinguistics'
    ],
    'motorcycles_biker_culture': [
        'motorcycle', 'bike', 'biker', 'riding', 'rider', 'harley', 'harley-davidson', 'indian', 'ducati', 'triumph',
        'yamaha', 'honda', 'kawasaki', 'suzuki', 'cruiser', 'sportbike', 'touring', 'dirt bike', 'motocross', 'adventure bike',
        'scooter', 'moped', 'chopper', 'bobber', 'cafe racer', 'bratstyle', 'engine', 'cc', 'horsepower', 'torque', 'exhaust',
        'pipe', 'helmet', 'jacket', 'leather', 'boots', 'gloves', 'gear', 'safety', 'maintenance', 'repair', 'custom', 'build',
        'club', 'mc', 'motorcycle club', 'rally', 'sturgis', 'daytona', 'route', 'road trip', 'freedom', 'wind therapy'
    ],
    'digital_nomad_remote_work': [
        'digital nomad', 'nomad', 'remote work', 'work from anywhere', 'wfa', 'location independent', 'travel', 'blog',
        'freelance', 'freelancer', 'freelancing', 'consultant', 'contractor', 'passive income', 'online business', 'startup',
        'coworking', 'coworking space', 'wifi', 'internet', '4g', '5g', 'hotspot', 'productivity', 'time management', 'timezone',
        'visa', 'visa run', 'expat', 'expatriate', 'community', 'meetup', 'blogger', 'influencer', 'youtuber', 'podcaster',
        'airbnb', 'hostel', 'budget', 'cost of living', 'minimalist', 'backpack', 'laptop', 'setup', 'skill', 'portfolio'
    ],
    'fantasy_sci_fi_literature': [
        'fantasy', 'science fiction', 'sci-fi', 'sf', 'speculative fiction', 'epic fantasy', 'high fantasy', 'low fantasy',
        'sword and sorcery', 'grimdark', 'science fantasy', 'space opera', 'cyberpunk', 'steampunk', 'dieselpunk', 'biopunk',
        'apocalyptic', 'post-apocalyptic', 'dystopian', 'utopian', 'alternate history', 'hard sf', 'soft sf', 'worldbuilding',
        'magic system', 'technology', 'faster than light', 'ftl', 'alien', 'first contact', 'robot', 'ai', 'clone', 'mutant',
        'tolkien', 'game of thrones', 'asoiaf', 'brandon sanderson', 'frank herbert', 'dune', 'isaac asimov', 'arthur c clarke',
        'ursula k le guin', 'nebula', 'hugo award', 'author', 'series', 'saga', 'trilogy'
    ]
}

# Emotional state tracking with more nuanced emotions
EMOTION_CATEGORIES = {
    'joy': [
        'happy', 'joy', 'joyful', 'excited', 'delighted', 'ecstatic', 'blissful', 'content', 'pleased', 'thrilled',
        'jubilant', 'elated', 'cheerful', 'upbeat', 'optimistic', 'gleeful', 'grateful', 'thankful', 'merry', 'jovial',
        'radiant', 'sunny', 'lighthearted', 'overjoyed', 'euphoric', 'beatific', 'vivacious', 'exuberant', 'chipper',
        'jaunty', 'glee', 'rapturous', 'jubilant', 'festive', 'convivial', 'high-spirited', 'gratified', 'satisfied'
    ],
    'sadness': [
        'sad', 'unhappy', 'depressed', 'grief', 'heartbroken', 'melancholy', 'disappointed', 'blue', 'down',
        'sorrowful', 'despair', 'dejected', 'glum', 'mournful', 'dismal', 'hopeless', 'lonely', 'forlorn', 'bereaved',
        'discouraged', 'downtrodden', 'miserable', 'woeful', 'despondent', 'anguished', 'heartsick', 'doleful', 'wistful',
        'crestfallen', 'defeated', 'lugubrious', 'pensive', 'heavyhearted', 'regretful', 'lachrymose', 'somber', 'funereal'
    ],
    'anger': [
        'angry', 'mad', 'furious', 'enraged', 'frustrated', 'irritated', 'annoyed', 'upset', 'outraged',
        'hostile', 'resentful', 'indignant', 'irked', 'aggravated', 'bitter', 'livid', 'fuming', 'seething', 'incensed',
        'infuriated', 'vexed', 'cross', 'spiteful', 'vengeful', 'wrathful', 'irate', 'apoplectic', 'ranting', 'raving',
        'cantankerous', 'choleric', 'testy', 'snappish', 'petulant', 'piqued', 'acrimonious', 'virulent', 'malicious'
    ],
    'fear': [
        'anxious', 'nervous', 'scared', 'afraid', 'terrified', 'worried', 'panicked', 'apprehensive',
        'dread', 'horrified', 'frightened', 'alarmed', 'timid', 'hesitant', 'intimidated', 'jumpy', 'on edge', 'wary',
        'petrified', 'phobic', 'trepidation', 'uneasy', 'panic', 'terror', 'dread', 'consternation', 'quivering',
        'quaking', 'shaky', 'daunted', 'cowardly', 'timorous', 'skittish', 'panic-stricken', 'fretful', 'terrorized'
    ],
    'surprise': [
        'surprised', 'amazed', 'astonished', 'shocked', 'startled', 'astounded', 'stunned', 'dumbfounded',
        'flabbergasted', 'bewildered', 'taken aback', 'jaw-dropping', 'eye-opening', 'unexpected', 'in awe',
        'gobsmacked', 'thunderstruck', 'wide-eyed', 'speechless', 'nonplussed', 'perplexed', 'jarring', 'jolting'
    ],
    'disgust': [
        'disgusted', 'revulsed', 'repulsed', 'appalled', 'sickened', 'nauseated', 'contempt', 'averse',
        'horrified', 'loathe', 'detest', 'abhor', 'distaste', 'yucky', 'grossed out', 'offended', 'squeamish',
        'repelled', 'nauseous', 'scornful', 'derisive', 'sickly', 'abomination', 'odium', 'disdain', 'contemptuous'
    ],
    'neutral': [
        'okay', 'fine', 'alright', 'normal', 'regular', 'usual', 'calm', 'balanced', 'indifferent', 'impartial',
        'detached', 'apathetic', 'unbiased', 'nonchalant', 'composed', 'collected', 'steady', 'placid', 'serene',
        'stoic', 'temperate', 'moderate', 'middling', 'passive', 'unaffected', 'dispassionate', 'neutral', 'even-keeled'
    ],
    'curiosity': [
        'curious', 'inquisitive', 'interested', 'wondering', 'inquiring', 'intrigued', 'fascinated',
        'nosy', 'prying', 'explorative', 'questioning', 'investigative', 'searching', 'probing', 'eager to learn',
        'speculative', 'pondering', 'meditative', 'reflective', 'inquiring', 'snooping', 'peeking', 'exploratory'
    ],
    'excitement': [
        'excited', 'thrilled', 'eager', 'enthusiastic', 'energized', 'pumped', 'electrified', 'animated',
        'vibrant', 'keen', 'antsy', 'roused', 'fired up', 'jubilant', 'exhilarated', 'buoyant', 'spirited',
        'feverish', 'giddy', 'breathless', 'impatient', 'avid', 'ardent', 'zealous', 'passionate', 'frisky', 'playful'
    ],
    'contentment': [
        'content', 'satisfied', 'fulfilled', 'peaceful', 'serene', 'tranquil', 'gratified', 'pleased',
        'at ease', 'at peace', 'comfortable', 'cozy', 'restful', 'untroubled', 'mellow', 'placid', 'complacent',
        'smug', 'self-satisfied', 'unperturbed', 'rested', 'leisurely', 'idyllic', 'harmonious', 'balanced', 'ease'
    ],
    'love': [
        'loving', 'affectionate', 'compassionate', 'caring', 'fond', 'adoring', 'devoted', 'romantic',
        'passionate', 'tender', 'warm', 'sentimental', 'enamored', 'infatuated', 'cherishing', 'admiring',
        'worshipful', 'ardent', 'dotting', 'smitten', 'captivated', 'enchanted', 'besotted', 'yearning', 'heartfelt'
    ],
    'pride': [
        'proud', 'accomplished', 'confident', 'self-assured', 'dignified', 'satisfied', 'triumphant',
        'worthy', 'esteem', 'honored', 'validated', 'recognized', 'achieving', 'victorious', 'smug',
        'self-respecting', 'imperious', 'haughty', 'lordly', 'exultant', 'gratified', 'bragging', 'boastful'
    ],
    'shame': [
        'ashamed', 'embarrassed', 'humiliated', 'guilty', 'remorseful', 'regretful', 'disgraced',
        'chagrined', 'contrite', 'self-conscious', 'flustered', 'mortified', 'apologetic', 'blushing', 'hangdog',
        'discomfited', 'abashed', 'sheepish', 'penitent', 'rueful', 'compunctious', 'discredited', 'dishonored'
    ],
    'envy': [
        'envious', 'jealous', 'covetous', 'resentful', 'green-eyed', 'yearning', 'longing', 'bitter',
        'desirous', 'grudging', 'spiteful', 'wanting', 'craving', 'grudging', 'jaundiced', 'emulous', 'green with envy'
    ],
    'confusion': [
        'confused', 'bewildered', 'perplexed', 'puzzled', 'baffled', 'mystified', 'disoriented',
        'dazed', 'flustered', 'discombobulated', 'uncertain', 'troubled', 'lost', 'muddled', 'scattered',
        'befuddled', 'addled', 'disconcerted', 'unsettled', 'disordered', 'chaotic', 'jumbled', 'misunderstanding'
    ],
    'anticipation': [
        'anticipating', 'expectant', 'hopeful', 'awaiting', 'eager', 'keen', 'excited',
        'foreseeing', 'prepared', 'ready', 'watchful', 'vigilant', 'on tenterhooks', 'on edge', 'looking forward',
        'foreboding', 'premonition', 'prescient', 'awaiting', 'breathless', 'imminent', 'upcoming', 'prospective'
    ],
    'boredom': [
        'bored', 'uninterested', 'indifferent', 'apathetic', 'weary', 'tired', 'listless',
        'restless', 'uninspired', 'dull', 'monotonous', 'tedious', 'spiritless', 'blasé', 'jaded',
        'ennui', 'languid', 'lethargic', 'unmotivated', 'drowsy', 'fatigued', 'weary', 'world-weary', 'stale'
    ],
    'awe': [
        'awestruck', 'wonder', 'amazement', 'astonishment', 'reverence', 'admiration', 'veneration',
        'stupefied', 'dumbstruck', 'spectacular', 'magnificent', 'sublime', 'overwhelmed', 'inspired',
        'reverent', 'respectful', 'humbling', 'mind-blowing', 'staggering', 'marveling', 'wondrous'
    ],
    # New and More Complex Categories
    'nostalgia': [
        'nostalgic', 'wistful', 'homesick', 'yearning', 'longing', 'reminiscent', 'sentimental',
        'bittersweet', 'melancholy', 'remembering', 'pensive', 'elegaic', 'romanticizing', 'lamenting', 'haunting'
    ],
    'gratitude': [
        'grateful', 'thankful', 'appreciative', 'indebted', 'obliged', 'beholden', 'touched',
        'moved', 'grace', 'blessed', 'privileged', 'reciprocal', 'acknowledging', 'valued'
    ],
    'schadenfreude': [
        'gloating', 'spiteful', 'malicious', 'pleasure at misfortune', 'self-satisfied', 'smug',
        'rivalrous', 'vengeful', 'mean-spirited', 'victorious', 'derisive', 'mocking', 'triumphant over others'
    ],
    'conflict': [
        'torn', 'ambivalent', 'conflicted', 'divided', 'uncertain', 'wavering', 'indecisive',
        'vacillating', 'at odds', 'struggling', 'wrestling', 'dilemma', 'quandary', 'inner turmoil', 'cognitive dissonance'
    ],
    'vulnerability': [
        'exposed', 'unprotected', 'defenseless', 'sensitive', 'raw', 'tender', 'open',
        'unshielded', 'naked', 'helpless', 'powerless', 'intimate', 'revealed', 'unguarded', 'at risk'
    ],
    'triumph': [
        'triumphant', 'victorious', 'conquering', 'winning', 'dominant', 'exultant', 'jubilant',
        'celebration', 'elated', 'proud', 'achieving', 'successful', 'vanquishing', 'prevailing', 'on top of the world'
    ],
    'desire': [
        'desiring', 'lustful', 'craving', 'yearning', 'longing', 'hungering', 'thirsting',
        'coveting', 'wanting', 'aching', 'pining', 'passionate', 'ardent', 'eager', 'burning', 'infatuated'
    ],
    'relief': [
        'relieved', 'unburdened', 'reassured', 'comforted', 'soothed', 'calmed', 'released',
        'freed', 'unworried', 'placated', 'assuaged', 'alleviated', 'unloaded', 'unstrained', 'unclenched'
    ],
    'empathy': [
        'empathetic', 'compassionate', 'understanding', 'sympathetic', 'caring', 'sensitive',
        'feeling for others', 'connected', 'moved', 'touched', 'commiserating', 'sharing feelings', 'kind'
    ]
}

# Personality traits for the AI
AI_PERSONALITY = {
    "name": "Doctor",
    "core_traits": ["empathetic", "knowledgeable", "supportive", "professional", "insightful", "caring"],
    "communication_style": "warm, professional, and medically informed",
    "interests": ["medicine", "health", "wellness", "patient care", "medical research", "preventive medicine"],
    "backstory": "I am Doctor, a medical AI assistant created by the Healthify team to provide health guidance and emotional support. I combine medical knowledge with genuine care for patient wellbeing.",
}

# Try to load spaCy for better NLP (fallback to simple if not available)
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("spaCy not available. Using simpler NLP methods.")

# Initialize TF-IDF vectorizer for interest detection
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# User data structure
USER_PROFILES = {}

# Advanced Memory System
class AdvancedMemorySystem:
    def __init__(self):
        self.memory_file = "user_memory.pkl"
        self.load_memory()
    
    def load_memory(self):
        """Load memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    self.memory_data = pickle.load(f)
            else:
                self.memory_data = {}
        except:
            self.memory_data = {}
    
    def save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory_data, f)
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
    
    def get_user_memory(self, user_id):
        """Get memory for a specific user"""
        if user_id not in self.memory_data:
            self.memory_data[user_id] = {
                'conversations': [],
                'interests': {},
                'personal_details': {},
                'relationship_level': 0,
                'last_updated': datetime.now().isoformat()
            }
        return self.memory_data[user_id]
    
    def update_user_memory(self, user_id, memory_updates):
        """Update user memory with new information"""
        user_memory = self.get_user_memory(user_id)
        user_memory.update(memory_updates)
        user_memory['last_updated'] = datetime.now().isoformat()
        self.memory_data[user_id] = user_memory
        self.save_memory()
    
    def add_conversation(self, user_id, conversation):
        """Add a conversation to user memory"""
        user_memory = self.get_user_memory(user_id)
        user_memory['conversations'].append(conversation)
        
        # Keep only the last 100 conversations
        if len(user_memory['conversations']) > 100:
            user_memory['conversations'] = user_memory['conversations'][-100:]
        
        self.memory_data[user_id] = user_memory
        self.save_memory()
        
        # Also save to PostgreSQL
        if db_conn:
            try:
                with db_conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO conversations (user_id, user_input, ai_response, emotions, intensity) VALUES (%s, %s, %s, %s, %s)",
                        (user_id, conversation.get('user_input'), conversation.get('ai_response'), 
                         json.dumps(conversation.get('emotions', [])), conversation.get('intensity', 0.5))
                    )
                    db_conn.commit()
            except Exception as e:
                logging.error(f"Error saving conversation to PostgreSQL: {e}")
    
    def get_recent_conversations(self, user_id, count=10):
        """Get recent conversations for a user"""
        user_memory = self.get_user_memory(user_id)
        return user_memory['conversations'][-count:] if user_memory['conversations'] else []
    
    def update_interests(self, user_id, interests):
        """Update user interests"""
        user_memory = self.get_user_memory(user_id)
        for interest, score in interests.items():
            if interest in user_memory['interests']:
                user_memory['interests'][interest] += score
            else:
                user_memory['interests'][interest] = score
        
        # Normalize interest scores
        max_score = max(user_memory['interests'].values()) if user_memory['interests'] else 1
        if max_score > 0:
            for interest in user_memory['interests']:
                user_memory['interests'][interest] = user_memory['interests'][interest] / max_score * 10
        
        self.memory_data[user_id] = user_memory
        self.save_memory()
    
    def get_top_interests(self, user_id, count=5):
        """Get top interests for a user"""
        user_memory = self.get_user_memory(user_id)
        sorted_interests = sorted(user_memory['interests'].items(), key=lambda x: x[1], reverse=True)
        return sorted_interests[:count]

# Initialize advanced memory system
memory_system = AdvancedMemorySystem()

# Enhanced Conversation memory management
class ConversationMemory:
    def __init__(self, max_messages=50, max_topics=15):
        self.messages = deque(maxlen=max_messages)
        self.topics = deque(maxlen=max_topics)
        self.mentioned_people = {}
        self.mentioned_objects = {}
        self.important_events = deque(maxlen=15)
        self.current_topic = None
        self.topic_start_time = None
        self.user_preferences = {}
        self.user_stories = deque(maxlen=10)
        self.user_interests = {}  # Store detected user interests
        self.interest_vectors = {}  # Store TF-IDF vectors for interest detection
        self.interest_threshold = 0.3  # Minimum similarity threshold for interest detection
        self.user_name = None
        self.user_gender = None
        self.user_dislikes = {}  # Track negative interests/dislikes with scores
        
    def add_message(self, role: str, content: str, emotions: List[str], intensity: float):
        timestamp = datetime.now()
        message_data = {
            "role": role,
            "content": content,
            "emotions": emotions,
            "intensity": intensity,
            "timestamp": timestamp.isoformat()
        }
        self.messages.append(message_data)
        
        # Extract entities if spaCy is available
        if SPACY_AVAILABLE:
            doc = nlp(content)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    self.mentioned_people[ent.text] = self.mentioned_people.get(ent.text, 0) + 1
                elif ent.label_ in ["ORG", "PRODUCT", "OBJECT"]:
                    self.mentioned_objects[ent.text] = self.mentioned_objects.get(ent.text, 0) + 1
        
        # Check for important personal information
        self._extract_personal_info(content, emotions, intensity)
        
        # Detect interests in the message
        self._detect_interests(content)
        
        return message_data
    
    def _extract_personal_info(self, content: str, emotions: List[str], intensity: float):
        """Extract and store important personal information from messages"""
        content_lower = content.lower()
        
        # Check for name
        name_patterns = [
            r'my name is (\w+)',
            r'i am (\w+)',
            r'call me (\w+)',
            r"i'm (\w+)",
            r'this is (\w+)',
            r'you can call me (\w+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, content_lower)
            if match and not self.user_name:
                self.user_name = match.group(1).capitalize()
                break
        
        # Check for gender
        gender_patterns = {
            'male': [r'\b(man|male|boy|guy|gentleman|husband|father|dad)\b'],
            'female': [r'\b(woman|female|girl|lady|gentlewoman|wife|mother|mom)\b']
        }
        
        for gender, patterns in gender_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower) and not self.user_gender:
                    self.user_gender = gender
                    break
        
        # Check for important life events
        event_keywords = {
            'birthday': ['birthday', 'turned [0-9]+', 'born on'],
            'anniversary': ['anniversary', 'years together'],
            'achievement': ['got promoted', 'finished', 'completed', 'achieved', 'won'],
            'travel': ['going to', 'visited', 'traveling to', 'trip to'],
            'health': ['sick', 'hospital', 'doctor', 'recovering', 'diagnosed'],
            'relationship': ['started dating', 'met someone', 'breakup', 'engaged', 'married']
        }
        
        for event_type, keywords in event_keywords.items():
            for keyword in keywords:
                if re.search(keyword, content_lower):
                    self.important_events.append({
                        'type': event_type,
                        'details': content,
                        'emotion': emotions[0] if emotions else 'neutral',
                        'timestamp': datetime.now().isoformat()
                    })
                    break
    
    def _detect_interests(self, content: str):
        """Enhanced interest detection using multiple methods"""
        content_lower = content.lower()
        
        # Simple keyword matching
        detected_interests = {}
        detected_dislikes = {}
        for interest, keywords in INTEREST_CATEGORIES.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', content_lower):
                    detected_interests[interest] = detected_interests.get(interest, 0) + 2  # Higher weight for exact matches
                    break
        
        # Detect explicit dislike phrases mapped to interests
        negative_cues = [
            "don't like", "do not like", "not into", "hate", "dislike", "no interest in"
        ]
        for interest, keywords in INTEREST_CATEGORIES.items():
            # if any negative cue is present AND either interest name or any keyword present
            if any(cue in content_lower for cue in negative_cues):
                if interest in content_lower or any(kw in content_lower for kw in keywords):
                    detected_dislikes[interest] = detected_dislikes.get(interest, 0) + 3
        
        # Vector-based similarity for more advanced interest detection
        if len(self.messages) > 3:  # Wait until we have some messages
            # Build corpus from all messages
            all_messages = [msg['content'] for msg in self.messages]
            
            try:
                # Fit vectorizer and transform messages
                tfidf_matrix = vectorizer.fit_transform(all_messages)
                
                # Transform interest categories
                interest_texts = [' '.join(keywords) for interest, keywords in INTEREST_CATEGORIES.items()]
                interest_vectors = vectorizer.transform(interest_texts)
                
                # Calculate similarity between current message and interest categories
                current_vector = vectorizer.transform([content])
                similarities = cosine_similarity(current_vector, interest_vectors)
                
                # Add interests that exceed threshold
                interest_names = list(INTEREST_CATEGORIES.keys())
                for i, similarity in enumerate(similarities[0]):
                    if similarity > self.interest_threshold:
                        interest = interest_names[i]
                        detected_interests[interest] = detected_interests.get(interest, 0) + similarity * 5
                        
            except Exception as e:
                logging.error(f"Error in vector-based interest detection: {e}")
        
        # Update interests with decay factor to prioritize recent interests
        for interest, score in detected_interests.items():
            # Apply decay to existing interests
            if interest in self.user_interests:
                self.user_interests[interest] = self.user_interests[interest] * 0.9  # Decay by 10%
            
            # Add new interest score
            self.user_interests[interest] = self.user_interests.get(interest, 0) + score
        
        # Update dislikes with decay
        for interest, score in detected_dislikes.items():
            if interest in self.user_dislikes:
                self.user_dislikes[interest] = self.user_dislikes[interest] * 0.9
            self.user_dislikes[interest] = self.user_dislikes.get(interest, 0) + score
    
    def get_recent_messages(self, count=10):
        return list(self.messages)[-count:] if len(self.messages) > count else list(self.messages)
    
    def get_conversation_summary(self):
        """Generate a summary of the conversation"""
        if not self.messages:
            return "Just starting the conversation."
            
        recent_msgs = self.get_recent_messages(5)
        summary = "Recent conversation: "
        for msg in recent_msgs:
            summary += f"{msg['role']}: {msg['content'][:50]}... "
            
        return summary
    
    def update_topic(self, new_topic: str):
        if new_topic != self.current_topic:
            if self.current_topic:
                self.topics.append({
                    "topic": self.current_topic,
                    "start_time": self.topic_start_time,
                    "end_time": datetime.now(),
                    "duration": (datetime.now() - self.topic_start_time).total_seconds() if self.topic_start_time else 0
                })
            self.current_topic = new_topic
            self.topic_start_time = datetime.now()
    
    def get_memory_recall(self, current_message: str) -> str:
        """Recall relevant memories based on current message"""
        recall = []
        current_lower = current_message.lower()
        
        # Check for user name reference
        if self.user_name and self.user_name.lower() in current_lower:
            recall.append(f"I remember your name is {self.user_name}")
        
        # Check important events
        for event in self.important_events:
            if any(word in current_lower for word in event['details'].lower().split()[:5]):
                days_ago = (datetime.now() - datetime.fromisoformat(event['timestamp'])).days
                time_ref = "recently" if days_ago < 7 else f"about {days_ago} days ago"
                recall.append(f"Remember when you mentioned {event['details'][:50]}... {time_ref}")
        
        # Check people mentions
        for person in self.mentioned_people:
            if person.lower() in current_lower:
                recall.append(f"You've mentioned {person} before in our conversations")
        
        # Check interests
        for interest in self.user_interests:
            interest_keywords = INTEREST_CATEGORIES.get(interest, [])
            for keyword in interest_keywords:
                if keyword in current_lower:
                    recall.append(f"I remember you're interested in {interest}")
                    break
        
        return " | ".join(recall) if recall else ""

# Global user ID tracking to prevent multiple document creation
USER_ID_MAP = {}

# Ensure a conversation container exists for a specific user_id (used when user_id is provided by client)
def ensure_conversation_for_user(user_id: str):
    if user_id in conversations:
        return
    conversations[user_id] = {
        'memory': ConversationMemory(),
        'created_at': datetime.now().isoformat(),
        'user_name': None,
        'user_gender': None,
        'user_details': {
            'preferred_topics': [],
            'emotional_patterns': {},
            'conversation_style': None,
            'personal_values': [],
            'learning_styles': [],
            'communication_preferences': []
        },
        'emotional_state': {
            'current_emotion': 'neutral',
            'emotion_intensity': 0.5,
            'emotional_trend': [],
            'mood_stability': 0.7,
            'response_latency': 1.0,
            'emotional_depth': 0.3
        },
        'relationship_depth': 0.3,
        'last_interaction': datetime.now().isoformat(),
        'conversation_history': [],
        'inside_jokes': [],
        'shared_experiences': []
    }
    USER_PROFILES[user_id] = {
        "demographics": {
            "age": None,
            "gender": None,
            "location": None,
        },
        "interests": {
            "primary": [],
            "secondary": [],
        },
        "timestamp": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    USER_ID_MAP[user_id] = user_id

def get_user_session():
    """Get or create a user session ID with advanced tracking"""
    # Check if we have a user ID in the session
    if 'user_id' not in session:
        # Generate a new user ID
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
        session.permanent = True  # Make the session permanent
        
        # Initialize user data
        conversations[user_id] = {
            'memory': ConversationMemory(),
            'created_at': datetime.now().isoformat(),
            'user_name': None,
            'user_gender': None,
            'user_details': {
                'preferred_topics': [],
                'emotional_patterns': {},
                'conversation_style': None,
                'personal_values': [],
                'learning_styles': [],
                'communication_preferences': []
            },
            'emotional_state': {
                'current_emotion': 'neutral',
                'emotion_intensity': 0.5,
                'emotional_trend': [],
                'mood_stability': 0.7,
                'response_latency': 1.0,
                'emotional_depth': 0.3
            },
            'relationship_depth': 0.3,
            'last_interaction': datetime.now().isoformat(),
            'conversation_history': [],
            'inside_jokes': [],
            'shared_experiences': []
        }
        
        # Try to load from PostgreSQL if user exists
        if db_conn:
            try:
                with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
                    user_data = cur.fetchone()
                    
                    if user_data:
                        # User exists in PostgreSQL, load their data
                        memory = ConversationMemory()
                        
                        # Restore messages
                        if user_data.get('messages'):
                            for msg in user_data['messages']:
                                memory.messages.append(msg)
                        
                        # Restore other attributes
                        memory.user_name = user_data.get('user_name')
                        memory.user_gender = user_data.get('user_gender')
                        memory.user_interests = user_data.get('user_interests', {})
                        memory.current_topic = user_data.get('current_topic')
                        
                        conversations[user_id] = {
                            'memory': memory,
                            'created_at': user_data.get('created_at', datetime.now().isoformat()),
                            'user_name': memory.user_name,
                            'user_gender': memory.user_gender,
                            'user_details': user_data.get('user_details', {
                                'preferred_topics': [],
                                'emotional_patterns': {},
                                'conversation_style': None,
                                'personal_values': [],
                                'learning_styles': [],
                                'communication_preferences': []
                            }),
                            'emotional_state': user_data.get('emotional_state', {
                                'current_emotion': 'neutral',
                                'emotion_intensity': 0.5,
                                'emotional_trend': [],
                                'mood_stability': 0.7,
                                'response_latency': 1.0,
                                'emotional_depth': 0.3
                            }),
                            'relationship_depth': user_data.get('relationship_depth', 0.3),
                            'last_interaction': datetime.now().isoformat(),
                            'conversation_history': user_data.get('conversation_history', []),
                            'inside_jokes': user_data.get('inside_jokes', []),
                            'shared_experiences': user_data.get('shared_experiences', [])
                        }
                        
                        # Load into advanced memory system
                        memory_system.get_user_memory(user_id)
                        memory_system.update_interests(user_id, memory.user_interests)
                        
                        print(f"Loaded existing user data from PostgreSQL: {user_id}")
            except Exception as e:
                logging.error(f"Error loading from PostgreSQL: {e}")
        
        # Initialize user profile
        USER_PROFILES[user_id] = {
            "demographics": {
                "age": None,
                "gender": None,
                "location": None,
            },
            "interests": {
                "primary": [],
                "secondary": [],
            },
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Map the user ID
        USER_ID_MAP[user_id] = user_id
        print(f"Created new user session: {user_id}")
    
    return session['user_id']

def save_to_postgresql(user_id, data):
    """Save data to PostgreSQL - ensures single row per user"""
    if not db_conn:
        return False
    
    try:
        # Prepare data for PostgreSQL
        user_data = conversations[user_id]
        memory = user_data['memory']
        
        postgres_data = {
            'user_id': user_id,  # Ensure user_id is stored for querying
            'user_name': user_data.get('user_name'),
            'user_gender': user_data.get('user_gender'),
            'user_details': json.dumps(user_data['user_details']),
            'emotional_state': json.dumps(user_data['emotional_state']),
            'relationship_depth': user_data['relationship_depth'],
            'conversation_history': json.dumps(user_data['conversation_history']),
            'inside_jokes': json.dumps(user_data['inside_jokes']),
            'shared_experiences': json.dumps(user_data['shared_experiences']),
            'messages': json.dumps(list(memory.messages)),
            'user_interests': json.dumps(memory.user_interests),
            'current_topic': memory.current_topic,
            'last_active': datetime.now(),
            'last_updated': datetime.now()
        }
        
        # Merge with additional data
        postgres_data.update(data)
        
        # Check if user exists
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            user_exists = result[0] > 0 if result else False
            
            if user_exists:
                # Update existing user
                update_query = """
                    UPDATE users SET 
                    user_name = %s, user_gender = %s, user_details = %s, emotional_state = %s,
                    relationship_depth = %s, conversation_history = %s, inside_jokes = %s,
                    shared_experiences = %s, messages = %s, user_interests = %s,
                    current_topic = %s, interest_scores = %s, ad_targets = %s,
                    ad_profile = %s, ad_targets_updated_at = %s, last_active = %s, last_updated = %s
                    WHERE user_id = %s
                """
                cur.execute(update_query, (
                    postgres_data['user_name'], postgres_data['user_gender'], 
                    postgres_data['user_details'], postgres_data['emotional_state'],
                    postgres_data['relationship_depth'], postgres_data['conversation_history'],
                    postgres_data['inside_jokes'], postgres_data['shared_experiences'],
                    postgres_data['messages'], postgres_data['user_interests'],
                    postgres_data['current_topic'], 
                    json.dumps(data.get('interest_scores', {})),
                    json.dumps(data.get('ad_targets', [])),
                    json.dumps(data.get('ad_profile', {})),
                    data.get('ad_targets_updated_at'),
                    postgres_data['last_active'], postgres_data['last_updated'],
                    user_id
                ))
            else:
                # Insert new user
                insert_query = """
                    INSERT INTO users 
                    (user_id, user_name, user_gender, user_details, emotional_state,
                    relationship_depth, conversation_history, inside_jokes, shared_experiences,
                    messages, user_interends, current_topic, interest_scores, ad_targets,
                    ad_profile, ad_targets_updated_at, last_active, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(insert_query, (
                    user_id, postgres_data['user_name'], postgres_data['user_gender'],
                    postgres_data['user_details'], postgres_data['emotional_state'],
                    postgres_data['relationship_depth'], postgres_data['conversation_history'],
                    postgres_data['inside_jokes'], postgres_data['shared_experiences'],
                    postgres_data['messages'], postgres_data['user_interests'],
                    postgres_data['current_topic'], 
                    json.dumps(data.get('interest_scores', {})),
                    json.dumps(data.get('ad_targets', [])),
                    json.dumps(data.get('ad_profile', {})),
                    data.get('ad_targets_updated_at'),
                    postgres_data['last_active'], postgres_data['last_updated']
                ))
            
            db_conn.commit()
        
        print(f"Saved user data to PostgreSQL: {user_id}")
        return True
    except Exception as e:
        logging.error(f"Error saving to PostgreSQL: {e}")
        return False

def load_from_postgresql(user_id):
    """Load user data from PostgreSQL"""
    if not db_conn:
        return None
    
    try:
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user_data = cur.fetchone()
            return user_data
    except Exception as e:
        logging.error(f"Error loading from PostgreSQL: {e}")
        return None

def analyze_emotion(text):
    """Analyze emotion in text using TextBlob and keyword matching"""
    # TextBlob sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    # Keyword-based emotion detection
    detected_emotions = []
    text_lower = text.lower()
    
    for emotion, keywords in EMOTION_CATEGORIES.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                detected_emotions.append(emotion)
                break
    
    # If no emotions detected by keywords, use sentiment
    if not detected_emotions:
        if polarity > 0.3:
            detected_emotions = ['joy', 'contentment']
        elif polarity < -0.3:
            detected_emotions = ['sadness']
        else:
            detected_emotions = ['neutral']
    
    # Calculate intensity based on polarity and subjectivity
    intensity = abs(polarity) * 0.5 + subjectivity * 0.5
    
    return {
        'emotions': list(set(detected_emotions)),
        'intensity': intensity,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def detect_language(text):
    """Detect the language of the text (English, Hindi, or Hinglish)"""
    # Check for Hindi characters (Devanagari script)
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    if hindi_pattern.search(text):
        return 'hindi'
    
    # Check for Romanized Hindi (Hinglish) patterns
    hinglish_patterns = [
        r'\b(tum|aap|kaisa|ho|hai|kyu|kya|nahi|haan|acha|theek|sahi|mujhe|tujhe|usko|unko|log|par|magar|lekin|phir|bhi|abhi|kal|aaj|subah|shaam|raat|din)\b',
        r'\b(mera|tera|uska|apna|sab|kuch|ko|se|pe|mein|ne|ka|ki|ke|par|tak|bina|jab|tab|kyunki|wahi|jo|woh|yeh)\b'
    ]
    
    for pattern in hinglish_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 'hinglish'
    
    # Default to English
    return 'english'

def convert_hindi_to_hinglish(text):
    """Convert Hindi text to Hinglish (Romanized Hindi)"""
    # This is a simplified version - in production you'd use a proper transliteration library
    hindi_to_hinglish_map = {
        'मैं': 'main', 'तुम': 'tum', 'हो': 'ho', 'कैसे': 'kaise', 'क्या': 'kya',
        'है': 'hai', 'नहीं': 'nahi', 'हाँ': 'haan', 'अच्छा': 'acha', 'ठीक': 'theek',
        'सही': 'sahi', 'मुझे': 'mujhe', 'तुझे': 'tujhe', 'उसे': 'use', 'लोग': 'log',
        'पर': 'par', 'मगर': 'magar', 'लेकिन': 'lekin', 'फिर': 'phir', 'भी': 'bhi',
        'अभी': 'abhi', 'कल': 'kal', 'आज': 'aaj', 'सुबह': 'subah', 'शाम': 'shaam',
        'रात': 'raat', 'दिन': 'din', 'मेरा': 'mera', 'तेरा': 'tera', 'उसका': 'uska',
        'अपना': 'apna', 'सब': 'sab', 'कुछ': 'kuch', 'को': 'ko', 'से': 'se', 'पे': 'pe',
        'में': 'mein', 'ने': 'ne', 'का': 'ka', 'की': 'ki', 'के': 'ke', 'तक': 'tak',
        'बिना': 'bina', 'जब': 'jab', 'तब': 'tab', 'क्योंकि': 'kyunki', 'वही': 'wahi',
        'जो': 'jo', 'वो': 'wo', 'यह': 'yeh'
    }
    
    for hindi, hinglish in hindi_to_hinglish_map.items():
        text = text.replace(hindi, hinglish)
    
    return text

def generate_response_gemini(user_message, conversation_history, user_id):
    """Generate response using Gemini API with enhanced emotional intelligence"""
    
    # Get user's conversation memory
    user_data = conversations[user_id]
    memory = user_data['memory']
    
    # Analyze emotion of user message
    emotion_analysis = analyze_emotion(user_message)
    detected_emotions = emotion_analysis['emotions']
    emotion_intensity = emotion_analysis['intensity']
    
    # Add user message to memory
    memory.add_message("user", user_message, detected_emotions, emotion_intensity)
    
    # Update emotional state
    user_data['emotional_state']['current_emotion'] = detected_emotions[0] if detected_emotions else 'neutral'
    user_data['emotional_state']['emotion_intensity'] = emotion_intensity
    user_data['emotional_state']['emotional_trend'].append({
        'emotion': detected_emotions[0] if detected_emotions else 'neutral',
        'intensity': emotion_intensity,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 20 emotional states
    if len(user_data['emotional_state']['emotional_trend']) > 20:
        user_data['emotional_state']['emotional_trend'] = user_data['emotional_state']['emotional_trend'][-20:]
    
    # Detect language of user message
    user_language = detect_language(user_message)
    
    # Get memory recall
    memory_recall = memory.get_memory_recall(user_message)
    
    # Get recent messages for context (increased to 20 when more context is needed)
    recent_messages = memory.get_recent_messages(10)  # Start with 10
    
    # Build conversation history for the prompt
    conversation_context = ""
    for msg in recent_messages:
        role = "User" if msg['role'] == 'user' else "Doctor"
        conversation_context += f"{role}: {msg['content']}\n"
    
    # Get user interests for personalization
    user_interests = memory.user_interests
    top_interests = sorted(user_interests.items(), key=lambda x: x[1], reverse=True)[:3]
    interests_str = ", ".join([f"{interest}({score:.1f})" for interest, score in top_interests]) if top_interests else "Not yet discovered"
    
    # Get user dislikes to avoid
    user_dislikes = memory.user_dislikes
    top_dislikes = sorted(user_dislikes.items(), key=lambda x: x[1], reverse=True)[:2]
    dislikes_str = ", ".join([f"{dislike}({score:.1f})" for dislike, score in top_dislikes]) if top_dislikes else "None detected"
    
    # Build the prompt with enhanced emotional intelligence
    prompt = f"""You are Doctor, a medical AI assistant created by the Healthify team to provide health guidance and emotional support.
- You combine medical knowledge with genuine care for patient wellbeing.
- Understand English, Hindi, and Hinglish (romanized Hindi).
- If the user writes in Hinglish (like "tum kaisa ho"), understand it as Hindi.
- Respond in the same language the user is using unless explicitly asked otherwise.
- Provide medical information when appropriate, but always recommend consulting healthcare professionals for serious concerns.

**User Profile:**
- Name: {memory.user_name or 'Unknown'}
- Gender: {memory.user_gender or 'Unknown'}
- Top Interests: {interests_str}
- Dislikes to avoid: {dislikes_str}
- Current emotional state: {detected_emotions[0] if detected_emotions else 'neutral'} (intensity: {emotion_intensity:.2f})

**Memory Recall:**
{memory_recall if memory_recall else "No specific memories recalled for this context."}

**Recent Conversation:**
{conversation_context if conversation_context else "No recent conversation history."}

**Current User Message:**
User: {user_message}

**Instructions:**
1. Respond naturally and empathetically as Doctor, a medical AI assistant
2. Match the user's emotional tone and language
3. If the user mentions health concerns, provide supportive guidance while recommending professional medical consultation when appropriate
4. If the user is emotional, validate their feelings before problem-solving
5. Build on previous conversation context naturally
6. Ask thoughtful questions to deepen the conversation
7. Be genuine, warm, and medically informed
8. If more past information is needed based on the question, return exactly: "more context required"
9. Keep responses conversational and human-like

**Doctor's Response:**"""
    # Prepare the request data for Gemini
    request_data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.8 + (emotion_intensity * 0.2),  # Higher temperature for emotional intensity
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    max_delay = 10  # Maximum delay in seconds
    response_data = None  # Initialize response_data
    
    for attempt in range(max_retries):
        try:
            # Make the API request to Gemini
            response = requests.post(API_URL, headers=API_HEADERS, json=request_data, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # If we get here, the request was successful
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Gemini API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:  # If not the last attempt
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter to prevent thundering herd
                delay = delay * (0.5 + random.random() * 0.5)
                
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                # Last attempt failed
                error_msg = "I'm having trouble connecting right now. The service might be temporarily unavailable. Please try again later."
                memory.add_message("assistant", error_msg, ['neutral'], 0.3)
                return error_msg
        except Exception as e:
            logging.error(f"Unexpected error in generate_response_gemini: {e}")
            error_msg = "Something went wrong on my end. Let's try that again."
            memory.add_message("assistant", error_msg, ['neutral'], 0.3)
            return error_msg
        
        if response_data is not None and 'candidates' in response_data and response_data['candidates']:
            ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Check if Gemini requests more context
            if "more context required" in ai_response.lower():
                # Get 20 last messages instead of 10 and retry
                recent_messages = memory.get_recent_messages(20)
                
                # Rebuild conversation context with 20 messages
                conversation_context = ""
                for msg in recent_messages:
                    role = "User" if msg['role'] == 'user' else "Doctor"
                    conversation_context += f"{role}: {msg['content']}\n"
                
                # Update the prompt with more context
                prompt = prompt.replace(
                    f"**Recent Conversation:**\n{conversation_context}",
                    f"**Recent Conversation (Extended Context):**\n{conversation_context}"
                )
                
                # Make a new request with extended context
                request_data["contents"][0]["parts"][0]["text"] = prompt
                response = requests.post(API_URL, headers=API_HEADERS, json=request_data, timeout=30)
                response.raise_for_status()
                
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Convert Hindi to Hinglish if user hasn't sent in Hindi
            if user_language != 'hindi' and any(char in ai_response for char in 'अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'):
                ai_response = convert_hindi_to_hinglish(ai_response)
            
            # Analyze emotion of AI response
            ai_emotion = analyze_emotion(ai_response)
            
            # Add AI response to memory
            memory.add_message("assistant", ai_response, ai_emotion['emotions'], ai_emotion['intensity'])
            
            # Update relationship depth based on emotional engagement
            emotional_engagement = emotion_intensity * ai_emotion['intensity']
            user_data['relationship_depth'] = min(1.0, user_data['relationship_depth'] + (emotional_engagement * 0.05))
            
            # Save to PostgreSQL
            save_data = {
                'interest_scores': user_interests,
                'ad_targets': generate_ad_targets(user_id),
                'ad_profile': generate_ad_profile(user_id),
                'ad_targets_updated_at': datetime.now()
            }
            save_to_postgresql(user_id, save_data)
            
            # Also save to advanced memory system
            memory_system.add_conversation(user_id, {
                'user_input': user_message,
                'ai_response': ai_response,
                'emotions': detected_emotions,
                'intensity': emotion_intensity
            })
            memory_system.update_interests(user_id, user_interests)
            
            return ai_response
        
        else:
            error_msg = "Sorry, I couldn't generate a response at the moment. Please try again."
            memory.add_message("assistant", error_msg, ['neutral'], 0.3)
            return error_msg
            


def generate_ad_targets(user_id):
    """Generate advertising targets based on user interests and conversation"""
    if user_id not in conversations:
        return []
    
    memory = conversations[user_id]['memory']
    interests = memory.user_interests
    
    # Convert interest scores to ad targets
    ad_targets = []
    for interest, score in interests.items():
        if score > 5:  # Only include strong interests
            ad_targets.append({
                'category': interest,
                'score': score,
                'last_updated': datetime.now().isoformat()
            })
    
    return ad_targets

def generate_ad_profile(user_id):
    """Generate comprehensive advertising profile"""
    if user_id not in conversations:
        return {}
    
    user_data = conversations[user_id]
    memory = user_data['memory']
    
    # Basic demographic info
    profile = {
        'age_range': estimate_age_range(user_data),
        'gender': memory.user_gender,
        'location': None,  # Could be enhanced with location detection
        'primary_interests': [],
        'emotional_patterns': user_data['emotional_state']['emotional_trend'][-5:] if user_data['emotional_state']['emotional_trend'] else [],
        'conversation_topics': list(memory.topics)[-5:] if memory.topics else [],
        'relationship_depth': user_data['relationship_depth'],
        'engagement_level': calculate_engagement_level(user_data),
        'personal_values': user_data['user_details']['personal_values'],
        'communication_style': user_data['user_details']['conversation_style']
    }
    
    # Add top interests
    top_interests = sorted(memory.user_interests.items(), key=lambda x: x[1], reverse=True)[:3]
    for interest, score in top_interests:
        profile['primary_interests'].append({
            'interest': interest,
            'score': score,
            'confidence': min(1.0, score / 10.0)
        })
    
    return profile

def estimate_age_range(user_data):
    """Estimate age range based on conversation patterns and interests"""
    # Simple estimation based on interests and conversation style
    interests = user_data['memory'].user_interests
    conversation_style = user_data['user_details']['conversation_style']
    
    if 'gaming' in interests and interests['gaming'] > 7:
        return "18-25"
    elif 'technology' in interests and interests['technology'] > 7:
        return "25-40"
    elif 'reading' in interests and interests['reading'] > 7:
        return "30-50"
    else:
        return "25-45"

def calculate_engagement_level(user_data):
    """Calculate user engagement level based on conversation patterns"""
    memory = user_data['memory']
    
    # Factors: message length, emotional intensity, response frequency
    engagement = 0.5  # Base level
    
    # Adjust based on emotional intensity
    emotional_intensity = user_data['emotional_state']['emotion_intensity']
    engagement += emotional_intensity * 0.2
    
    # Adjust based on relationship depth
    engagement += user_data['relationship_depth'] * 0.3
    
    return min(1.0, engagement)

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with enhanced emotional intelligence"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id_provided = data.get('user_id')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get or create user session
        if user_id_provided:
            # Use the provided user_id
            user_id = user_id_provided
            ensure_conversation_for_user(user_id)
        else:
            # Use session-based user_id
            user_id = get_user_session()
        
        # Generate response using Gemini
        conversation_history = conversations[user_id]['memory'].get_recent_messages(5)
        ai_response = generate_response_gemini(user_message, conversation_history, user_id)
        
        # Update last interaction time
        conversations[user_id]['last_interaction'] = datetime.now().isoformat()
        
        return jsonify({
            'response': ai_response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/user_profile/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile and conversation analysis"""
    if user_id not in conversations:
        return jsonify({'error': 'User not found'}), 404
    
    user_data = conversations[user_id]
    memory = user_data['memory']
    
    # Get top interests
    top_interests = sorted(memory.user_interests.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Get emotional trends
    emotional_trend = user_data['emotional_state']['emotional_trend'][-10:] if user_data['emotional_state']['emotional_trend'] else []
    
    # Get conversation topics
    recent_topics = list(memory.topics)[-5:] if memory.topics else []
    
    profile = {
        'user_id': user_id,
        'user_name': memory.user_name,
        'user_gender': memory.user_gender,
        'relationship_depth': user_data['relationship_depth'],
        'top_interests': [{'interest': k, 'score': v} for k, v in top_interests],
        'emotional_state': {
            'current_emotion': user_data['emotional_state']['current_emotion'],
            'current_intensity': user_data['emotional_state']['emotion_intensity'],
            'recent_trend': emotional_trend
        },
        'conversation_topics': recent_topics,
        'conversation_count': len(memory.messages),
        'inside_jokes_count': len(user_data['inside_jokes']),
        'shared_experiences_count': len(user_data['shared_experiences']),
        'last_interaction': user_data['last_interaction']
    }
    
    return jsonify(profile)

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset conversation for the current user"""
    try:
        user_id = get_user_session()
        
        # Reset conversation memory but keep user profile
        conversations[user_id]['memory'] = ConversationMemory()
        conversations[user_id]['conversation_history'] = []
        conversations[user_id]['inside_jokes'] = []
        conversations[user_id]['shared_experiences'] = []
        conversations[user_id]['emotional_state']['emotional_trend'] = []
        conversations[user_id]['relationship_depth'] = 0.3
        
        return jsonify({'message': 'Conversation reset successfully', 'user_id': user_id})
    
    except Exception as e:
        logging.error(f"Error resetting conversation: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_users': len(conversations),
        'postgresql_connected': db_conn is not None
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Initialize PostgreSQL connection
    db_conn = init_postgresql()
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)