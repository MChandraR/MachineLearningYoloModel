import paho.mqtt.client as mqtt

# Callback saat terhubung ke broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("esp32cheat/notif")

# Callback saat menerima pesan
def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(username="emqx",password="public")

# Ganti '192.168.1.100' dengan IP broker MQTT Anda
client.connect("broker.mqtt.cool", 1883, 60)

client.loop_forever()
