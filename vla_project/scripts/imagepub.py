import redis 
import cv2
import threading
from queue import Queue, Empty
import time
"""                        SOME KEY IDEAS 
Sleep mechanism in case of redis connection failure

publish_loop
    └── Failed completely (sleep here)
        └── connect_redis
            └── Individual attempt failed (sleep here)
            └── Next attempt
            └── Individual attempt failed (sleep here)
            └── etc...

ImagePublisher (Redis Client)               Redis Server            Subscribers
┌────────────────────────────┐            ┌──────────────┐         ┌────────────┐
│                            │            │              │         │            │
│  ┌────────────┐            │            │              │         │            │
│  │Frame Queue │            │   Connect  │              │         │            │
│  │(maxsize=1) ├──────┐     │◄──────────►│              │         │            │
│  └────────────┘      │     │            │              │         │            │
│                      ▼     │            │              │         │            │
│  ┌────────────────────┐    │            │  ┌────────┐  │         │ Your Ros   |     
│  │   publish_loop     │    │  Publish   │  │Channel │  │  Sub    │ Node lives |          
│  │   (Thread)         ├─── ┼──────────► │  │"image_ │  ├────────►│ here       │
│  └────────────────────┘    │            │  │channel"│  │         │            │
│          ▲                 │            │  └────────┘  │         │            │
│          │                 │            │              │         │            │
│  ┌───────┴──────────┐      │            │              │         │            │
│  │  connect_redis   │      │            │              │         │            │
│  │  (with retries)  │      │            │              │         │            │
│  └──────────────────┘      │            │              │         │            │
│                            │            │              │         │            │
└────────────────────────────┘            └──────────────┘         └────────────┘

Data Flow:
1. Frame → Queue (publish_frame method)
2. publish_loop thread:
   - Dequeues frame
   - Converts to JPEG
   - Adds metadata (width,height)
   - Publishes to Redis

Connection Features:
- Socket keepalive enabled
- Connection timeout: 5s
- Auto-retry on timeout
- Max retries: 5
- Retry delay: 1s

Queue Features:
- Max size: 1 frame
- Drops old frame if full
- Non-blocking put/get
"""

class ImagePublisher:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.publish_thread = None
        self.connection_retry_delay = 1
        self.max_retries = 5

    # connect_redis: 
    # 1. Connects to Redis
    # 2. Returns True if successful
    def connect_redis(self):
        retries = 0
        while retries < self.max_retries and self.running:
            try:
                if self.redis_client is None or not self.redis_client.ping():
                    print(f"Attempting to connect to Redis (attempt {retries + 1}/{self.max_retries})...")
                    self.redis_client = redis.Redis(
                        host=self.redis_host, 
                        port=self.redis_port,
                        socket_keepalive=True,
                        socket_connect_timeout=5,
                        retry_on_timeout=True
                    )
                    self.redis_client.ping()
                    print("Successfully connected to Redis!")  
                return True
            
            except redis.ConnectionError as e:
                print(f"Redis connection failed: {e}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.connection_retry_delay} seconds...")
                    time.sleep(self.connection_retry_delay)
                self.redis_client = None
                
            except Exception as e:
                print(f"Unexpected error while connecting to Redis: {e}")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.connection_retry_delay)
                self.redis_client = None
                
        return False    

    # publish_loop: 
    # 1. Dequeues frame
    # 2. Converts to JPEG
    # 3. Adds metadata (width,height)
    # 4. Publishes to Redis
    def publish_loop(self):
        while self.running:
            try:
                if not self.connect_redis():
                    print("Failed to connect to Redis after multiple attempts")
                    time.sleep(self.connection_retry_delay)
                    continue

                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except Empty:
                    continue

                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    continue

                img_bytes = buffer.tobytes()
                height, width = frame.shape[:2]
                metadata = f"{width},{height}|".encode()

                try:
                    self.redis_client.publish("image_channel", metadata + img_bytes)
                except redis.ConnectionError as e:
                    print(f"Lost connection to Redis: {e}")
                    # force reconnection 
                    self.redis_client = None
                    continue
                except Exception as e:
                    print(f"Error publishing frame: {e}")
                    continue 
                # small delay to prevent processing overload
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in publish loop: {e}")
                continue
    # Start
    # 1. Sets running to True
    # 2. Creates publish thread
    # 3. Starts publish thread
    def start(self):
        if not self.running:
            self.running = True
            self.publish_thread = threading.Thread(target=self.publish_loop)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            print("Publisher started")
    
    def stop(self):
        self.running = False
        if self.publish_thread:
            # we need to wait for the thread to finish before closing the redis client
            self.publish_thread.join(timeout=5.0) 
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        print("Publisher stopped")
    # publish_frame
    # 1. Checks if frame is None
    # 2. Checks if frame queue is full
    #    - If full, drops old frame
    # 3. Puts frame in queue
    def publish_frame(self, frame):
        if frame is None:
            return
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait() 
                except Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except Exception as e:
            print(f"Error queuing frame: {e}")