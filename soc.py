import streamlit as st
import socketio
import asyncio
import cv2
import base64
import numpy as np
from aiohttp import web

import socketio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.event
async def my_message(data):
    print('message received with ')
    
@sio.event
async def get_video(data):
    print('Menerima data video ')

@sio.event
async def disconnect():
    print('disconnected from server')

async def main():
    await sio.connect('http://localhost:3000')
    await sio.wait()

@sio.event
async def frame_data(data):
    print("frame ")
    jpg_original = base64.b64decode(data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Update Streamlit image slot
    image_slot.image(img_rgb, use_column_width=True)


# Streamlit configuration
st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="centered")
st.title("Streamlit Image Processing Challenge SIC 5")
st.write("""
### Unggah gambar Anda dan pilih efek pengolahan gambar:
""")
image_slot = st.empty()
st.video("tokyo.mp4", format="video/mp4", start_time=10, subtitles=None, end_time=None, loop=False, autoplay=True, muted=False)
if __name__ == '__main__':
    asyncio.run(main())
