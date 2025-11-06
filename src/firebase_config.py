"""
Firebase Configuration
- Initializes the Firebase Admin SDK
- Provides a function to get a Firestore client
"""

import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import json

# --- Firebase Admin SDK Initialization ---
# The service account key is expected to be in a JSON file
# specified by the GOOGLE_APPLICATION_CREDENTIALS environment variable.

def initialize_firebase():
    """
    Initializes the Firebase Admin SDK using service account credentials.
    """
    if not firebase_admin._apps:
        try:
            # Use a service account
            # The GOOGLE_APPLICATION_CREDENTIALS env var should point to your JSON key file
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
            })
            print("Firebase initialized successfully.")
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            # Fallback for local development if you want to use a local key
            # key_path = "path/to/your/serviceAccountKey.json"
            # if os.path.exists(key_path):
            #     cred = credentials.Certificate(key_path)
            #     firebase_admin.initialize_app(cred)
            # else:
            #     print("GOOGLE_APPLICATION_CREDENTIALS not set and local key not found.")


def get_firestore_client():
    """
    Returns a Firestore client instance.
    """
    initialize_firebase()
    return firestore.client()

def get_storage_bucket():
    """
    Returns a storage bucket instance.
    """
    initialize_firebase()
    return storage.bucket()
