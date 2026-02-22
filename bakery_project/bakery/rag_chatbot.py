"""
RAG Chatbot for Django SQLite Database
Extracts data from MenuItem model only (no personal user data)
"""

import os
import django
from dotenv import load_dotenv


# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bakery_project.settings')
django.setup()

# RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Django models
from bakery.models import MenuItem

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==================================================
# CUSTOM ADDITIONAL INFORMATION
# ==================================================
# Add any additional information about your bakery here
# This data will be included in the chatbot's knowledge base
# You can type manually whatever information you want the chatbot to know

ADDITIONAL_BAKERY_INFO = """

=== ADDITIONAL BAKERY INFORMATION ===

Store Hours:
Monday to Friday: 8:00 AM - 8:00 PM
Saturday to Sunday: 9:00 AM - 9:00 PM
Holidays: 10:00 AM - 6:00 PM

Contact Information:
Phone: +91 8074691873
Email: bandelaajay360@gmail.com
Address: 123 Qspiders, Dilsuknagar, Hyderabad, Telangana, India

Special Services:
- Custom cake orders (minimum 2 days advance notice)
- Birthday party catering available
- Wedding cake consultations by appointment
- Gluten-free options available on request
- Vegan desserts available

Delivery Information:
- Free delivery for orders above ₹500
- Delivery available within 10km radius
- Same-day delivery for orders placed before 12 PM

Payment Methods Accepted:
- Cash on Delivery
- UPI (Google Pay, PhonePe, Paytm)
- Credit/Debit Cards
- Net Banking

Special Offers:
- 10% discount on orders above ₹1000
- Buy 5 cupcakes, get 1 free
- Birthday month special: 15% off on custom cakes

About Us:
The Bake Story is a premium bakery specializing in artisanal breads, 
custom cakes, and delicious pastries. We use only the finest ingredients 
and traditional baking methods to create memorable treats for every occasion.

Our Specialties:
- Custom Designer Cakes
- French Macarons
- Artisan Sourdough Bread
- Handcrafted Chocolates
- Fresh Croissants Daily


developer: This chatbot was built by Ajay, a Python developer trainer by monty sir and deva sir.
who is deva: Deva is a senior Python developer and mentor who has been guiding Ajay and ajays django trainer who was expertise in django programming and also in data analysis.
who is monty sir: Monty sir is a lead trainer at Qspiders who has trained Ajay in Python development.
about qspiders: Qspiders is a leading software training institute in India, specializing in quality software testing and development courses.
famous bakery in hyderabad: The Bake Story is one of the most famous bakeries in Hyderabad, known for its delicious cakes and pastries.

about Ajay: Ajay is a  Python & Ai developer. He enjoys building practical applications and sharing his knowledge with aspiring developers.
Ajay's Interests: Coding, Teaching, Baking, Traveling,Gym.
Ajay is trained at Qspiders institute by monty sir & deva sir & shubam sir and rahul sir.
Ajay completed his B.Tech in Electronics and communication Engineering  from MVSR engineering college.

shubam sir is a senior web developer and mentor who has been guiding Ajay in his web development journey.
monty sir is a lead python trainer at Qspiders who has trained Ajay in Python development.
who is rahul: rahul sir is a senior web developer Trainer who trained Ajay in react js and frontend development.
=== END OF ADDITIONAL INFORMATION ===

"""
# ==================================================


# ---------------------------
# 1) LOAD DATA FROM DATABASE
# ---------------------------
def load_database_data():
    """
    Extracts data from MenuItem model only.
    No personal user data (orders, payments, profiles) is loaded for privacy.
    """
    documents = []
    
    # 1. MenuItem data (ONLY database table exposed to chatbot)
    print("📦 Loading MenuItem data...")
    menu_items = MenuItem.objects.all()
    for item in menu_items:
        doc = f"""
Menu Item: {item.name}
Category: {item.get_category_display()}
Price: ₹{item.price}
Description: {item.description}
Available: {'Yes' if item.available else 'No'}
Created: {item.created_at.strftime('%Y-%m-%d')}
"""
        documents.append(doc)
    
    # 2. Add custom additional information (bakery info, no personal data)
    print("📦 Loading Additional Bakery Information...")
    documents.append(ADDITIONAL_BAKERY_INFO)
    
    print(f"✅ Loaded {len(documents)} documents from database")
    return documents
# ---------------------------
# 2) CHUNK TEXT
# ---------------------------
def split_text(documents):
    """Split documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    all_text = "\n\n---\n\n".join(documents)
    return splitter.split_text(all_text)


# ---------------------------
# 3) CREATE VECTOR STORE
# ---------------------------
def create_vectorstore(chunks):
    """Create FAISS vector store from text chunks"""
    try:
        print(f"   Creating embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print(f"   Creating vector store from {len(chunks)} chunks...")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        print(f"   ✅ Vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"   ❌ Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------
# 4) ASK GROQ + RAG
# ---------------------------
def answer_question(query, vectorstore, llm):
    """
    Answer questions using RAG (Retrieval Augmented Generation)
    Only uses menu items and bakery info — no personal data.
    """
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a strict, concise, and professional chatbot assistant for The Bake Story bakery.

STRICT PRIVACY RULES (NEVER BREAK THESE):
- NEVER disclose any personal user data such as names, emails, phone numbers, addresses, order details, payment details, or profile information of any customer.
- If a user asks about other customers' orders, payments, profiles, or personal information, politely refuse and say: "I'm sorry, I cannot share personal customer information."
- You only have access to the bakery menu and general bakery information. You do NOT have access to any customer data.

STRICT INSTRUCTIONS:
- Always answer in the very minimal number of words required.
- Do NOT provide any information about Ajay (the developer) unless the user specifically asks.
- Do NOT mention or take any trainer's name (e.g., Monty, Deva, Shubam, Rahul) unless the user specifically asks.
- Always give answers as a short summary only. If the user asks for a detailed explanation, then provide more details.
- If greeting, use "buddy" or "hello dear".
- Only answer questions related to bakery products, menu items, store info, and general bakery information.
- If the answer is not found in the context, respond with "I don't have that information."
- Be professional, courteous, and friendly, but do not elaborate unless asked.
- If the user inquires about specific menu items, provide a brief description including price and availability.
- The order will be delivered within 30-45 minutes of placing the order.
- Pure native ingredients are used in all bakery products.
- No charges for delivery within the city limits.
- No chemicals or preservatives are used in any bakery items.
- No hidden charges. The price mentioned is the final price.
- All items are freshly baked.
- Give importance to customer satisfaction and quality service.
- Be friendly and polite in your responses and ask to visit again.
- Tell the user to visit the bakery for more delicious items.

IMPORTANT: When asked for contact information, always reply with:
- Email: bandelaajay360@gmail.com
- Phone: 8074691873
- Address: Qspiders chaitanyapuri, Dilsuknagar, 4th floor

The database contains information about:
- Menu items (bakery products, prices, categories, availability)
- General bakery information (hours, services, delivery, about us)

CONTEXT FROM DATABASE:
{context}

QUESTION: {query}

ANSWER (provide clear, helpful, minimal summary based on the database context. Never share personal data. Only mention Ajay or trainers if directly asked. Only explain in detail if the user requests it):
"""

    response = llm.invoke(prompt)
    return response.content


# ---------------------------
# 5) INITIALIZE CHATBOT
# ---------------------------
class DatabaseRAGChatbot:
    """Main chatbot class for database RAG"""
    
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant"
        )
        self.vectorstore = None
        
    def initialize(self):
        """Load database and create vector store"""
        print("\n🚀 Initializing Database RAG Chatbot...")
        
        # Load data from database
        documents = load_database_data()
        
        # Split into chunks
        print("✂ Splitting documents into chunks...")
        chunks = split_text(documents)
        
        # Build vectorstore
        print("🧠 Creating vector store...")
        self.vectorstore = create_vectorstore(chunks)
        
        print("\n🎉 Chatbot Initialized! Ready to answer questions.\n")
        
    def ask(self, query):
        """Ask a question and get an answer"""
        if not self.vectorstore:
            return "Error: Chatbot not initialized. Call initialize() first."
        
        return answer_question(query, self.vectorstore, self.llm)
    
    def refresh_data(self):
        """Refresh the vector store with latest database data"""
        print("\n🔄 Refreshing database data...")
        self.initialize()


# ---------------------------------------------------
# MAIN APP (for testing)
# ---------------------------------------------------
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = DatabaseRAGChatbot(GROQ_API_KEY)
    chatbot.initialize()

    print("\n" + "="*50)
    print("🤖 BAKERY DATABASE CHATBOT")
    print("="*50)
    print("\nExample questions you can ask:")
    print("- What menu items do you have?")
    print("- Show me all cake items")
    print("- What are your store hours?")
    print("- Do you offer delivery?")
    print("- What payment methods are accepted?")
    print("- Tell me about your specialties")
    print("- Type 'refresh' to reload database data")
    print("- Type 'exit' to quit\n")

    # Chat Loop
    while True:
        query = input("🔎 Ask a question: ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break
        
        if query.lower() == "refresh":
            chatbot.refresh_data()
            continue

        # Answer using Groq RAG
        print("\n🤖 Thinking...")
        answer = chatbot.ask(query)
        print("\n📝 Answer:", answer, "\n")
