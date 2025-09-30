"""
Collaborative Features and Sharing Capabilities
Real-time collaboration, sharing, and social features
"""

import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import base64
import streamlit as st
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ShareableSimulation:
    """Shareable simulation data structure."""
    id: str
    title: str
    description: str
    author: str
    created_at: str
    updated_at: str
    simulation_data: Dict[str, Any]
    tags: List[str]
    visibility: str  # 'public', 'private', 'unlisted'
    likes: int
    views: int
    forks: int
    is_featured: bool = False

@dataclass
class CollaborationSession:
    """Real-time collaboration session."""
    session_id: str
    participants: List[str]
    active_simulation: Optional[str]
    created_at: str
    last_activity: str
    permissions: Dict[str, List[str]]  # user_id -> ['read', 'write', 'admin']

class SimulationSharing:
    """Simulation sharing and collaboration system."""
    
    def __init__(self, data_dir: str = "data/shared"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for active sessions
        self.active_sessions = {}
        self.shared_simulations = {}
        self.user_profiles = {}
        
        # Load existing data
        self._load_shared_data()
    
    def _load_shared_data(self):
        """Load existing shared simulations and user data."""
        try:
            # Load shared simulations
            shared_file = self.data_dir / "shared_simulations.json"
            if shared_file.exists():
                with open(shared_file, 'r') as f:
                    self.shared_simulations = json.load(f)
            
            # Load user profiles
            profiles_file = self.data_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    self.user_profiles = json.load(f)
        except Exception as e:
            logger.error(f"Error loading shared data: {e}")
    
    def _save_shared_data(self):
        """Save shared simulations and user data."""
        try:
            # Save shared simulations
            shared_file = self.data_dir / "shared_simulations.json"
            with open(shared_file, 'w') as f:
                json.dump(self.shared_simulations, f, indent=2)
            
            # Save user profiles
            profiles_file = self.data_dir / "user_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving shared data: {e}")
    
    def create_shareable_simulation(self, simulation_data: Dict[str, Any], 
                                  title: str, description: str, author: str,
                                  tags: List[str] = None, visibility: str = "public") -> str:
        """Create a shareable simulation."""
        simulation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        shareable = ShareableSimulation(
            id=simulation_id,
            title=title,
            description=description,
            author=author,
            created_at=timestamp,
            updated_at=timestamp,
            simulation_data=simulation_data,
            tags=tags or [],
            visibility=visibility,
            likes=0,
            views=0,
            forks=0
        )
        
        self.shared_simulations[simulation_id] = asdict(shareable)
        self._save_shared_data()
        
        logger.info(f"Created shareable simulation: {simulation_id}")
        return simulation_id
    
    def get_shared_simulation(self, simulation_id: str) -> Optional[ShareableSimulation]:
        """Get a shared simulation by ID."""
        if simulation_id in self.shared_simulations:
            data = self.shared_simulations[simulation_id]
            return ShareableSimulation(**data)
        return None
    
    def list_shared_simulations(self, author: Optional[str] = None, 
                              tags: Optional[List[str]] = None,
                              visibility: str = "public") -> List[ShareableSimulation]:
        """List shared simulations with filters."""
        simulations = []
        
        for sim_data in self.shared_simulations.values():
            sim = ShareableSimulation(**sim_data)
            
            # Apply filters
            if author and sim.author != author:
                continue
            if sim.visibility != visibility:
                continue
            if tags and not any(tag in sim.tags for tag in tags):
                continue
            
            simulations.append(sim)
        
        # Sort by creation date (newest first)
        simulations.sort(key=lambda x: x.created_at, reverse=True)
        return simulations
    
    def like_simulation(self, simulation_id: str, user_id: str) -> bool:
        """Like a simulation."""
        if simulation_id in self.shared_simulations:
            self.shared_simulations[simulation_id]['likes'] += 1
            self._save_shared_data()
            return True
        return False
    
    def view_simulation(self, simulation_id: str) -> bool:
        """Record a simulation view."""
        if simulation_id in self.shared_simulations:
            self.shared_simulations[simulation_id]['views'] += 1
            self._save_shared_data()
            return True
        return False
    
    def fork_simulation(self, simulation_id: str, new_author: str, 
                       new_title: str = None) -> Optional[str]:
        """Fork (copy) a simulation."""
        original = self.get_shared_simulation(simulation_id)
        if not original:
            return None
        
        # Create new simulation based on original
        new_simulation_data = original.simulation_data.copy()
        new_title = new_title or f"Fork of {original.title}"
        
        new_id = self.create_shareable_simulation(
            simulation_data=new_simulation_data,
            title=new_title,
            description=f"Forked from: {original.title}",
            author=new_author,
            tags=original.tags.copy(),
            visibility="public"
        )
        
        # Increment fork count on original
        self.shared_simulations[simulation_id]['forks'] += 1
        self._save_shared_data()
        
        return new_id
    
    def create_collaboration_session(self, creator: str, 
                                   simulation_id: Optional[str] = None) -> str:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = CollaborationSession(
            session_id=session_id,
            participants=[creator],
            active_simulation=simulation_id,
            created_at=timestamp,
            last_activity=timestamp,
            permissions={creator: ['admin']}
        )
        
        self.active_sessions[session_id] = asdict(session)
        
        logger.info(f"Created collaboration session: {session_id}")
        return session_id
    
    def join_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """Join an existing collaboration session."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            if user_id not in session_data['participants']:
                session_data['participants'].append(user_id)
                session_data['permissions'][user_id] = ['read', 'write']
                session_data['last_activity'] = datetime.now().isoformat()
                return True
        return False
    
    def leave_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            if user_id in session_data['participants']:
                session_data['participants'].remove(user_id)
                if user_id in session_data['permissions']:
                    del session_data['permissions'][user_id]
                session_data['last_activity'] = datetime.now().isoformat()
                return True
        return False
    
    def get_collaboration_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session details."""
        if session_id in self.active_sessions:
            return CollaborationSession(**self.active_sessions[session_id])
        return None
    
    def list_active_sessions(self, user_id: Optional[str] = None) -> List[CollaborationSession]:
        """List active collaboration sessions."""
        sessions = []
        
        for session_data in self.active_sessions.values():
            session = CollaborationSession(**session_data)
            
            if user_id and user_id not in session.participants:
                continue
            
            sessions.append(session)
        
        return sessions
    
    def generate_share_link(self, simulation_id: str, 
                          link_type: str = "view") -> str:
        """Generate a shareable link for a simulation."""
        base_url = st.get_option("server.baseUrlPath") or "http://localhost:8501"
        
        if link_type == "view":
            return f"{base_url}/?simulation={simulation_id}"
        elif link_type == "fork":
            return f"{base_url}/?fork={simulation_id}"
        elif link_type == "collaborate":
            return f"{base_url}/?collaborate={simulation_id}"
        else:
            return f"{base_url}/?simulation={simulation_id}"
    
    def create_embed_code(self, simulation_id: str, 
                         width: int = 800, height: int = 600) -> str:
        """Create embed code for a simulation."""
        base_url = st.get_option("server.baseUrlPath") or "http://localhost:8501"
        
        embed_code = f"""
<iframe 
    src="{base_url}/?simulation={simulation_id}&embed=true"
    width="{width}" 
    height="{height}"
    frameborder="0"
    allowfullscreen>
</iframe>
"""
        return embed_code
    
    def export_simulation_package(self, simulation_id: str) -> str:
        """Export simulation as a complete package."""
        simulation = self.get_shared_simulation(simulation_id)
        if not simulation:
            return None
        
        # Create package directory
        package_dir = self.data_dir / "packages" / simulation_id
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create package manifest
        manifest = {
            "id": simulation.id,
            "title": simulation.title,
            "description": simulation.description,
            "author": simulation.author,
            "created_at": simulation.created_at,
            "version": "1.0",
            "files": [
                "simulation_data.json",
                "README.md",
                "requirements.txt"
            ]
        }
        
        # Save manifest
        with open(package_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save simulation data
        with open(package_dir / "simulation_data.json", 'w') as f:
            json.dump(simulation.simulation_data, f, indent=2)
        
        # Create README
        readme_content = f"""# {simulation.title}

{simulation.description}

## Author
{simulation.author}

## Created
{simulation.created_at}

## Tags
{', '.join(simulation.tags)}

## Usage
1. Import the simulation_data.json file into the Wave Theory Chatbot
2. Run the simulation with the provided parameters
3. Explore the physics and modify as needed

## Sharing
This simulation was shared from the Wave Theory Chatbot collaborative platform.
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        # Create requirements file
        requirements = """# Wave Theory Chatbot Requirements
streamlit>=1.28.2
plotly>=5.18.0
numpy>=1.24.3
pandas>=2.1.3
jax>=0.4.20
equinox>=0.11.2
pysr>=0.16.2
"""
        
        with open(package_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        # Create ZIP package
        import zipfile
        zip_path = package_dir.parent / f"{simulation_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(package_dir))
        
        return str(zip_path)
    
    def create_community_dashboard(self) -> Dict[str, Any]:
        """Create community dashboard with statistics."""
        total_simulations = len(self.shared_simulations)
        total_views = sum(sim['views'] for sim in self.shared_simulations.values())
        total_likes = sum(sim['likes'] for sim in self.shared_simulations.values())
        total_forks = sum(sim['forks'] for sim in self.shared_simulations.values())
        
        # Most popular simulations
        popular_simulations = sorted(
            self.shared_simulations.values(),
            key=lambda x: x['likes'] + x['views'],
            reverse=True
        )[:10]
        
        # Recent simulations
        recent_simulations = sorted(
            self.shared_simulations.values(),
            key=lambda x: x['created_at'],
            reverse=True
        )[:10]
        
        # Active collaboration sessions
        active_sessions = len(self.active_sessions)
        
        return {
            "total_simulations": total_simulations,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_forks": total_forks,
            "active_sessions": active_sessions,
            "popular_simulations": popular_simulations,
            "recent_simulations": recent_simulations
        }

class SocialFeatures:
    """Social features for the Wave Theory community."""
    
    def __init__(self, sharing_system: SimulationSharing):
        self.sharing = sharing_system
        self.comments = {}
        self.follows = {}
        self.notifications = {}
    
    def add_comment(self, simulation_id: str, user_id: str, 
                   comment: str, parent_id: Optional[str] = None) -> str:
        """Add a comment to a simulation."""
        comment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        comment_data = {
            "id": comment_id,
            "simulation_id": simulation_id,
            "user_id": user_id,
            "comment": comment,
            "parent_id": parent_id,
            "created_at": timestamp,
            "likes": 0
        }
        
        if simulation_id not in self.comments:
            self.comments[simulation_id] = []
        
        self.comments[simulation_id].append(comment_data)
        
        # Notify simulation author
        simulation = self.sharing.get_shared_simulation(simulation_id)
        if simulation:
            self._add_notification(
                simulation.author,
                f"New comment on '{simulation.title}'",
                comment
            )
        
        return comment_id
    
    def get_comments(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Get comments for a simulation."""
        return self.comments.get(simulation_id, [])
    
    def like_comment(self, simulation_id: str, comment_id: str) -> bool:
        """Like a comment."""
        if simulation_id in self.comments:
            for comment in self.comments[simulation_id]:
                if comment['id'] == comment_id:
                    comment['likes'] += 1
                    return True
        return False
    
    def follow_user(self, follower_id: str, target_id: str) -> bool:
        """Follow a user."""
        if follower_id not in self.follows:
            self.follows[follower_id] = []
        
        if target_id not in self.follows[follower_id]:
            self.follows[follower_id].append(target_id)
            return True
        return False
    
    def unfollow_user(self, follower_id: str, target_id: str) -> bool:
        """Unfollow a user."""
        if follower_id in self.follows and target_id in self.follows[follower_id]:
            self.follows[follower_id].remove(target_id)
            return True
        return False
    
    def get_followers(self, user_id: str) -> List[str]:
        """Get followers of a user."""
        followers = []
        for follower, following in self.follows.items():
            if user_id in following:
                followers.append(follower)
        return followers
    
    def get_following(self, user_id: str) -> List[str]:
        """Get users that a user is following."""
        return self.follows.get(user_id, [])
    
    def _add_notification(self, user_id: str, title: str, message: str):
        """Add a notification for a user."""
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        notification = {
            "id": str(uuid.uuid4()),
            "title": title,
            "message": message,
            "created_at": datetime.now().isoformat(),
            "read": False
        }
        
        self.notifications[user_id].append(notification)
    
    def get_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        user_notifications = self.notifications.get(user_id, [])
        
        if unread_only:
            return [n for n in user_notifications if not n['read']]
        
        return user_notifications
    
    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read."""
        if user_id in self.notifications:
            for notification in self.notifications[user_id]:
                if notification['id'] == notification_id:
                    notification['read'] = True
                    return True
        return False

# Global instances
simulation_sharing = SimulationSharing()
social_features = SocialFeatures(simulation_sharing)
