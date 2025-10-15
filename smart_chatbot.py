"""5-Star Hotel Customer Support Chatbot
Enhanced with images, rich content, and beautiful UI
"""

import os
import re
import json
import time
import logging
import requests
from typing import List, Dict, Any, Tuple
from PIL import Image
import base64
from io import BytesIO

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Enhanced Configuration
# ---------------------------
logging.basicConfig(
    filename="hotel_chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hotel_chatbot")

# Hotel branding
HOTEL_NAME = "GRAND HOTELS"
HOTEL_TAGLINE = "The Land of Luxury"

# ---------------------------
# Enhanced Synthetic Data with Images
# ---------------------------

# Sample images (URLs from Unsplash - free to use)
HOTEL_IMAGES = {
    "presidential_suite": "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=600",
    "deluxe_room": "https://images.unsplash.com/photo-1522771739844-6a9f6d5f14af?w=600",
    "rooftop_dining": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=600",
    "spa": "https://images.unsplash.com/photo-1544161515-4ab6ce6db874?w=600",
    "pool": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=600",
    "lobby": "https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=600",
    "wedding": "https://images.unsplash.com/photo-1511795409834-ef04bbd61622?w=600",
    "gym": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=600"
}

ENHANCED_SYNTHETIC_DOCS = [
    {
        "id": "room_presidential",
        "title": "Presidential Suite",
        "text": """
        The Presidential Suite at Grand Luxe Palace offers breathtaking ocean views from its private balcony. 
        This 2,500 sq ft suite features a king-size four-poster bed with Egyptian cotton linens, a living room with fireplace, 
        formal dining area, and a study. The marble bathroom includes a Jacuzzi tub, steam shower, and premium toiletries.
        
        Amenities: Private pool, 24/7 butler service, complimentary champagne on arrival, daily fruit basket, 
        Nespresso machine, smart room controls, and personalized concierge service.
        
        Capacity: Up to 4 guests
        Price: $2,500 per night
        Best for: Special occasions, business executives, luxury seekers
        """,
        "image": HOTEL_IMAGES["presidential_suite"],
        "category": "accommodation",
        "tags": ["luxury", "suite", "ocean-view", "butler-service"]
    },
    {
        "id": "room_deluxe",
        "title": "Deluxe Room",
        "text": """
        Our Deluxe Rooms offer stunning city views with elegant contemporary design. Each 450 sq ft room features 
        a queen-size bed with premium mattress, work desk with ergonomic chair, 55-inch smart TV, and high-speed Wi-Fi.
        
        Amenities: Turn-down service, mini-bar, coffee maker, ironing facilities, safe deposit box, and daily housekeeping.
        Complimentary access to fitness center and business lounge.
        
        Capacity: 2 guests
        Price: $450 per night
        Best for: Business travelers, couples, short stays
        """,
        "image": HOTEL_IMAGES["deluxe_room"],
        "category": "accommodation",
        "tags": ["standard", "city-view", "business", "comfort"]
    },
    {
        "id": "dining_rooftop",
        "title": "SkyView Rooftop Restaurant",
        "text": """
        Experience fine dining 30 floors above the city at our award-winning rooftop restaurant. 
        Executive Chef Marco Bellini creates innovative Mediterranean cuisine with locally sourced ingredients.
        
        Menu Highlights: Seafood platter, truffle pasta, wagyu beef, vegan tasting menu
        Operating Hours: Dinner 6PM-11PM, Bar 5PM-1AM
        Special Features: Live jazz music (7PM-10PM), private dining rooms, sommelier service
        Dress Code: Smart casual
        Reservation: Required, especially for weekend dining
        """,
        "image": HOTEL_IMAGES["rooftop_dining"],
        "category": "dining",
        "tags": ["fine-dining", "rooftop", "romantic", "bar"]
    },
    {
        "id": "spa_wellness",
        "title": "Serenity Spa & Wellness Center",
        "text": """
        Our 10,000 sq ft spa offers a sanctuary of relaxation and rejuvenation. 
        Treatments blend ancient techniques with modern wellness practices.
        
        Services: Swedish massage, deep tissue, hot stone therapy, aromatherapy, facials, body wraps
        Facilities: Steam room, sauna, Jacuzzi, hydrotherapy pool, relaxation lounge
        Special Packages: Couples retreat, detox program, executive stress relief
        Hours: 8AM-10PM daily
        Booking: Through concierge or mobile app, 24-hour advance notice recommended
        """,
        "image": HOTEL_IMAGES["spa"],
        "category": "wellness",
        "tags": ["spa", "massage", "relaxation", "wellness"]
    },
    {
        "id": "pool_facilities",
        "title": "Infinity Pool & Recreation",
        "text": """
        Our temperature-controlled infinity pool offers stunning panoramic views of the city skyline. 
        The pool area includes private cabanas, poolside service, and a dedicated children's pool.
        
        Facilities: Main pool (25 meters), heated Jacuzzi, sun deck, pool bar, towel service
        Hours: 6AM-10PM
        Services: Poolside dining, cocktail service, cabana rentals ($150/day)
        Rules: Children must be supervised, no outside food or drinks
        """,
        "image": HOTEL_IMAGES["pool"],
        "category": "facilities",
        "tags": ["pool", "recreation", "family", "relaxation"]
    },
    {
        "id": "events_wedding",
        "title": "Event & Wedding Services",
        "text": """
        Host unforgettable events in our elegant ballrooms and outdoor spaces. 
        Our dedicated event team ensures every detail is perfect.
        
        Venues: Grand Ballroom (300 guests), Garden Terrace (150 guests), Boardrooms (10-50 guests)
        Services: Catering, floral arrangements, AV equipment, event planning, photography
        Wedding Packages: From $15,000 including ceremony, reception, and honeymoon suite
        Corporate Events: Team building, conferences, product launches
        """,
        "image": HOTEL_IMAGES["wedding"],
        "category": "events",
        "tags": ["wedding", "events", "corporate", "celebration"]
    },
    {
        "id": "fitness_center",
        "title": "State-of-the-Art Fitness Center",
        "text": """
        Maintain your workout routine in our 24/7 fitness center equipped with latest Technogym equipment.
        
        Equipment: Treadmills, ellipticals, weight machines, free weights, yoga studio
        Classes: Morning yoga (7AM), pilates (6PM), personal training available
        Amenities: Towel service, water station, locker rooms with showers
        Access: Complimentary for all hotel guests
        """,
        "image": HOTEL_IMAGES["gym"],
        "category": "facilities",
        "tags": ["fitness", "gym", "wellness", "24/7"]
    },
    {
        "id": "policies_checkin",
        "title": "Check-in & Check-out Policies",
        "text": """
        Check-in Time: 3:00 PM
        Check-out Time: 12:00 PM
        Early Check-in: Subject to availability, complimentary before 11 AM, $100 fee for 11 AM-3 PM
        Late Check-out: Subject to availability, complimentary until 2 PM, $150 fee for 2 PM-6 PM
        
        Requirements: Valid ID, credit card for incidentals ($200 hold)
        Payment: All major credit cards, cash, mobile payments accepted
        Age Policy: Must be 21+ to check in
        """,
        "category": "policies",
        "tags": ["check-in", "policies", "procedures"]
    },
    {
        "id": "transportation",
        "title": "Transportation Services",
        "text": """
        Airport Transfer: Mercedes-Benz sedan ($75 one-way), SUV ($100 one-way)
        City Shuttle: Complimentary shuttle to downtown (every 30 minutes, 8AM-10PM)
        Car Rental: Partner discounts with Avis and Hertz
        Valet Parking: $45 per night with in/out privileges
        Taxi/Uber: Designated pickup area at main entrance
        """,
        "category": "services",
        "tags": ["transport", "airport", "parking", "shuttle"]
    }
]

# ---------------------------
# Enhanced UI Components
# ---------------------------

def load_image_from_url(url):
    """Load image from URL with error handling"""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        return None

def image_to_base64(image):
    """Convert PIL image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_service_card(title, description, image_url, price=None):
    """Create a beautiful service card"""
    try:
        image = load_image_from_url(image_url)
        if image:
            st.image(image, use_column_width=True)
    except:
        st.image("🏨", use_column_width=True)  # Fallback emoji
    
    st.subheader(title)
    st.write(description)
    if price:
        st.metric("Starting Price", price)
    st.markdown("---")

# ---------------------------
# Utility functions (updated)
# ---------------------------

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split a long text into overlapping chunks by characters."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

def enrich_docs(docs: List[Dict[str, str]]) -> List[Document]:
    """Convert synthetic doc dicts into langchain Documents with metadata."""
    documents = []
    for d in docs:
        text = d["text"]
        title = d.get("title", "")
        doc_id = d.get("id") or title
        category = d.get("category", "general")
        tags = d.get("tags", [])
        
        chunks = chunk_text(text)
        for i, chunk_text_content in enumerate(chunks):
            metadata = {
                "source_id": doc_id, 
                "title": title, 
                "chunk": i,
                "category": category,
                "tags": json.dumps(tags),
                "image": d.get("image", "")
            }
            documents.append(Document(page_content=chunk_text_content, metadata=metadata))
    return documents

# ... (Keep the existing guardrails, API key management, vector store functions from previous version)
# Guardrails and utility functions remain the same as in the previous working version

def sanitize_input(user_text: str) -> Tuple[bool, str]:
    """Sanitize and perform guardrail checks."""
    text = user_text.strip().lower()
    for bad in BLACKLIST:
        if bad in text:
            return False, f"Query blocked due to disallowed term: {bad}"
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, text):
            return False, "Query looks like a prompt-injection attempt and was blocked."
    if len(text) > 2000:
        return False, "Query too long. Please shorten your question."
    return True, user_text

def get_api_key():
    """Get API key from secrets, environment, or user input."""
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        if hasattr(st, 'secrets') and st.secrets is not None:
            if 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    if hasattr(st, 'session_state') and hasattr(st.session_state, 'get'):
        return st.session_state.get('api_key')
    return None

def get_embeddings_model():
    """Get free HuggingFace embeddings model."""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings model: {str(e)}")
        st.stop()

def build_vector_store(documents: List[Document]) -> FAISS:
    """Create or load FAISS vector store."""
    try:
        embeddings = get_embeddings_model()
        VDB_DIR = "faiss_store"
        index_file = os.path.join(VDB_DIR, "index.faiss")
        
        if os.path.exists(index_file):
            return FAISS.load_local(VDB_DIR, embeddings, allow_dangerous_deserialization=True)
        else:
            os.makedirs(VDB_DIR, exist_ok=True)
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]
            db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            db.save_local(VDB_DIR)
            return db
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        st.stop()

# ---------------------------
# Enhanced Response Generation
# ---------------------------

def format_response_with_images(answer: str, sources: List[Dict]) -> str:
    """Enhance response with image references and rich formatting."""
    enhanced_answer = answer
    
    # Add image references if available
    image_sources = [s for s in sources if s.get('image')]
    if image_sources:
        enhanced_answer += "\n\n📸 *Visual references available for this information*"
    
    # Add category tags
    categories = list(set([s.get('category', '') for s in sources if s.get('category')]))
    if categories:
        enhanced_answer += f"\n\n🏷️ Related: {', '.join(categories)}"
    
    return enhanced_answer

def get_fallback_response(query: str, retrieved_docs: List[Document]) -> str:
    """Generate enhanced fallback responses."""
    if not retrieved_docs:
        return f"🏨 I'd be happy to help you with information about {HOTEL_NAME}! Currently I'm operating in basic mode. For detailed assistance, please contact our front desk at extension 0."

    context = " ".join([doc.page_content for doc in retrieved_docs[:3]])
    query_lower = query.lower()

    # Enhanced keyword responses
    responses = {
        'room': f"✨ We offer luxurious accommodations including Presidential Suites ($2,500/night) and Deluxe Rooms ($450/night). All rooms feature premium amenities, stunning views, and 24/7 concierge service.",
        'suite': "👑 Our Presidential Suite offers 2,500 sq ft of luxury with private pool, butler service, and ocean views. Perfect for special occasions!",
        'dining': "🍽️ Experience award-winning dining at our SkyView Rooftop Restaurant with Mediterranean cuisine, live jazz, and panoramic city views. Reservations recommended!",
        'spa': "💆 Our Serenity Spa offers massages, facials, and wellness treatments in a 10,000 sq ft sanctuary. Open 8AM-10PM daily.",
        'pool': "🏊 Our infinity pool features stunning skyline views, private cabanas, and poolside service. Open 6AM-10PM.",
        'wedding': "💍 Host unforgettable weddings in our elegant ballrooms. Packages start at $15,000 including ceremony, reception, and honeymoon suite.",
        'gym': "💪 Our 24/7 fitness center features Technogym equipment, yoga classes, and personal training. Complimentary for all guests.",
        'check': "⏰ Check-in at 3PM, check-out at 12PM. Early check-in and late check-out available upon availability.",
        'price': "💰 Room rates start from $450/night for Deluxe Rooms. Contact reservations for special packages and discounts.",
        'transport': "🚗 Airport transfers available from $75. Complimentary downtown shuttle every 30 minutes."
    }

    for keyword, response in responses.items():
        if keyword in query_lower:
            return response

    return f"🏨 Based on our information: {context[:250]}... For specific details or to make a reservation, please contact our concierge team! 🌟"

# ---------------------------
# Enhanced Main Application
# ---------------------------

def main():
    # Page configuration with enhanced styling
    st.set_page_config(
        page_title=f"{HOTEL_NAME} - Luxury Hotel Assistant",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tagline {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border-left: 5px solid #3498db;
    }
    .user-message {
        background-color: #ecf0f1;
        border-left-color: #2ecc71;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #3498db;
    }
    .service-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .main-header {
    font-size: 3rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
}
.tagline {
    font-size: 1.2rem;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-container {
    background-color: #000000;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.chat-message {
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
    border-left: 4px solid #3498db;
}
.user-message {
    background-color: #2c3e50;
    border-left-color: #2ecc71;
    color: #ffffff;
}
.assistant-message {
    background-color: #1a1a1a;
    border-left-color: #3498db;
    color: #ffffff;
}
.service-card {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.stChatInput {
    background-color: #000000;
}
.stChatInput input {
    color: #ffffff;
    background-color: #2c3e50;
}
    </style>
    """, unsafe_allow_html=True)
# Header section with image above text (Simple and clean)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.container():
        # Image on top (centered)
        try:
            st.image("assets/Screenshot 2025-10-15 at 7.29.03 AM.png")
        except FileNotFoundError:
            st.markdown('<div style="text-align: center; font-size: 100px; margin-bottom: 20px;">🏨</div>', unsafe_allow_html=True)
        
        # Text below image (centered)
        st.markdown(f'<h1 class="main-header" style="text-align: center; margin: 20px 0 10px 0;">{HOTEL_NAME}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="tagline" style="text-align: center; margin: 0;">{HOTEL_TAGLINE}</p>', unsafe_allow_html=True)

# Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)



    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_db_ready' not in st.session_state:
        st.session_state.vector_db_ready = False
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'hybrid' not in st.session_state:
        st.session_state.hybrid = None

    # Sidebar with enhanced features
    with st.sidebar:
        st.image("assets/hotel-1979406_1280.jpg")
        
        st.header("🎯 Ask Me About These Services")
        if st.button("📋 Room Types"):
            st.session_state.messages.append({"role": "user", "content": "Show me room options"})
        if st.button("🍽️ Dining Information"):
            st.session_state.messages.append({"role": "user", "content": "Tell me about dining options"})
        if st.button("💆 Spa Services"):
            st.session_state.messages.append({"role": "user", "content": "What spa services do you offer?"})
        
    
        
        st.markdown("---")
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🏨 Hotel Services", "ℹ️ About Us"])

    with tab1:
        # Chat interface
        st.subheader("💬 How can I assist you today?")
        
        # Initialize system
        if not st.session_state.vector_db_ready:
            with st.spinner("🔄 Initializing luxury hotel assistant..."):
                try:
                    documents = enrich_docs(ENHANCED_SYNTHETIC_DOCS)
                    vdb = build_vector_store(documents)
                    
                    # Simplified retriever setup
                    class SimpleRetriever:
                        def __init__(self, vector_db):
                            self.vector_db = vector_db
                        def retrieve(self, query):
                            return self.vector_db.similarity_search(query, k=3)
                    
                    st.session_state.hybrid = SimpleRetriever(vdb)
                    st.session_state.vector_db_ready = True
                    st.success("✅ Luxury hotel assistant ready!")
                except Exception as e:
                    st.error(f"Initialization failed: {e}")

        # Display chat messages with enhanced styling
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🏨 **Assistant:** {msg["content"]}</div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask about our luxury services..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("💭 Thinking..."):
                # Simple response generation (replace with your RAG pipeline)
                if st.session_state.hybrid:
                    try:
                        results = st.session_state.hybrid.retrieve(prompt)
                        response = get_fallback_response(prompt, results)
                    except:
                        response = "I apologize, but I'm experiencing technical difficulties. Please contact our front desk for immediate assistance."
                else:
                    response = "Welcome to Grand Luxe Palace! I can help with room bookings, dining reservations, spa services, and more. How may I assist you today?"
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    with tab2:
        st.subheader("🌟 Our Luxury Services")
        
        # Service showcase
        cols = st.columns(2)
        services = [
            ("Luxury Accommodations", "Presidential Suites, Deluxe Rooms", HOTEL_IMAGES["presidential_suite"], "$450-$2,500/night"),
            ("Fine Dining", "Rooftop restaurant, 24/7 room service", HOTEL_IMAGES["rooftop_dining"], "$$$"),
            ("Spa & Wellness", "Massages, treatments, relaxation", HOTEL_IMAGES["spa"], "$$"),
            ("Event Spaces", "Weddings, conferences, celebrations", HOTEL_IMAGES["wedding"], "Custom pricing")
        ]
        
        for i, (title, desc, img, price) in enumerate(services):
            with cols[i % 2]:
                create_service_card(title, desc, img, price)

    with tab3:
        st.subheader("🏨 About Grand Luxe Palace")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(HOTEL_IMAGES["lobby"], use_column_width=True)
            st.write("""
            ### 🌟 Award-Winning Luxury
            - ★★★★★ 5-Star Rating
            - Travelers' Choice 2024
            - Luxury Hotel Awards 2023
            """)
        
        with col2:
            st.write("""
            ### 🎯 Our Amenities
            - 24/7 Concierge Service
            - Infinity Pool & Spa
            - Fine Dining Restaurants
            - State-of-the-Art Fitness Center
            - Business Center & Meeting Rooms
            - Complimentary Wi-Fi
            - Valet Parking
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p> Grand Hotels  • 123 Luxury Avenue • Premium City • +1-555-LUXE-HOTEL</p>
        <p>© 2025 Grand Hotels. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()