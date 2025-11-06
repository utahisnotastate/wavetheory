
import asyncio
import websockets
import json

class CortexClient:
    def __init__(self, client_id, client_secret, **kwargs):
        self.uri = "wss://localhost:6868"
        self.client_id = client_id
        self.client_secret = client_secret
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)

    async def send(self, message):
        await self.websocket.send(json.dumps(message))

    async def receive(self):
        return json.loads(await self.websocket.recv())

    async def authorize(self):
        auth_request = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "authorize",
            "params": {
                "clientId": self.client_id,
                "clientSecret": self.client_secret
            }
        }
        await self.send(auth_request)
        response = await self.receive()
        return response.get("result", {}).get("cortexToken")

    async def create_session(self, token, headset_id):
        session_request = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "createSession",
            "params": {
                "cortexToken": token,
                "headset": headset_id,
                "status": "active"
            }
        }
        await self.send(session_request)
        response = await self.receive()
        return response.get("result", {}).get("id")

    async def subscribe(self, token, session_id, streams):
        subscribe_request = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {
                "cortexToken": token,
                "session": session_id,
                "streams": streams
            }
        }
        await self.send(subscribe_request)
        await self.receive()  # Consume the subscription response

    async def close(self):
        await self.websocket.close()

async def use_cortex(client_id, client_secret, headset_id):
    client = CortexClient(client_id, client_secret)
    await client.connect()
    token = await client.authorize()
    session_id = await client.create_session(token, headset_id)
    await client.subscribe(token, session_id, ["eeg"])

    async for message in client.websocket:
        data = json.loads(message)
        if "eeg" in data:
            yield data["eeg"]

    await client.close()
