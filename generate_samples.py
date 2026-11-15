import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images(output_dir="PixelTalk/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pizza (Food Scene)
    img_pizza = Image.new("RGB", (640, 480), color=(235, 215, 185))
    draw = ImageDraw.Draw(img_pizza)
    # Wooden table planks
    for y in range(0, 480, 40):
        draw.line([(0, y), (640, y)], fill=(190, 160, 130), width=2)
    # Plate
    draw.ellipse([110, 30, 530, 450], fill=(245, 245, 250), outline=(210, 210, 215), width=4)
    # Pizza crust
    draw.ellipse([130, 50, 510, 430], fill=(210, 150, 75), outline=(160, 100, 45), width=6)
    # Pizza sauce & cheese
    draw.ellipse([145, 65, 495, 415], fill=(255, 210, 95))
    # Tomato sauce splotches
    draw.ellipse([180, 120, 260, 190], fill=(210, 45, 30))
    draw.ellipse([340, 240, 430, 320], fill=(210, 45, 30))
    draw.ellipse([250, 300, 330, 370], fill=(210, 45, 30))
    # Pepperoni / Tomatoes slices
    pepperonis = [(220, 140), (370, 150), (280, 230), (390, 280), (230, 330), (340, 350), (180, 240)]
    for px, py in pepperonis:
        draw.ellipse([px-30, py-30, px+30, py+30], fill=(195, 35, 30), outline=(140, 20, 15), width=2)
    # Basil leaves (green)
    basils = [(250, 190), (330, 190), (310, 280), (220, 280), (410, 220)]
    for bx, by in basils:
        draw.ellipse([bx-18, by-10, bx+18, by+10], fill=(45, 145, 55))
    # Text label banner
    draw.rectangle([20, 20, 210, 55], fill=(30, 30, 30, 200))
    draw.text((30, 28), "Fresh Italian Pizza", fill=(255, 255, 255))
    img_pizza.save(os.path.join(output_dir, "pizza.jpg"), quality=95)

    # 2. Dog in Park (Animal Scene)
    img_dog = Image.new("RGB", (640, 480), color=(140, 205, 250)) # Sky
    draw = ImageDraw.Draw(img_dog)
    # Sun
    draw.ellipse([500, 30, 580, 110], fill=(255, 220, 50))
    # Green Grass
    draw.rectangle([0, 240, 640, 480], fill=(60, 165, 60))
    draw.ellipse([-50, 200, 400, 350], fill=(70, 180, 70))
    draw.ellipse([300, 210, 700, 350], fill=(55, 155, 55))
    # Trees in background
    for tx in [80, 180, 480, 580]:
        draw.rectangle([tx+15, 170, tx+35, 260], fill=(110, 75, 45))
        draw.ellipse([tx-25, 90, tx+75, 190], fill=(35, 125, 45))
    # Dog body (Golden Retriever puppy)
    draw.ellipse([240, 260, 390, 370], fill=(225, 175, 95)) # torso
    draw.ellipse([350, 230, 440, 310], fill=(235, 185, 105)) # head
    draw.ellipse([410, 255, 435, 280], fill=(50, 40, 35)) # snout
    draw.ellipse([380, 245, 395, 260], fill=(25, 20, 15)) # eye
    # Dog ears
    draw.polygon([(360, 235), (345, 280), (375, 270)], fill=(200, 150, 75))
    # Dog paws
    draw.rectangle([250, 350, 275, 410], fill=(220, 170, 90))
    draw.rectangle([340, 350, 365, 410], fill=(220, 170, 90))
    # Red toy ball
    draw.ellipse([430, 360, 480, 410], fill=(230, 40, 40), outline=(170, 20, 20), width=2)
    draw.rectangle([20, 20, 230, 55], fill=(30, 30, 30))
    draw.text((30, 28), "Golden Retriever & Ball", fill=(255, 255, 255))
    img_dog.save(os.path.join(output_dir, "dog_park.jpg"), quality=95)

    # 3. Vintage Car (Vehicle / Street Scene)
    img_car = Image.new("RGB", (640, 480), color=(180, 210, 230))
    draw = ImageDraw.Draw(img_car)
    # City buildings in background
    draw.rectangle([40, 80, 180, 320], fill=(160, 170, 180))
    draw.rectangle([200, 50, 340, 320], fill=(140, 150, 165))
    draw.rectangle([360, 100, 520, 320], fill=(175, 180, 190))
    # Street road
    draw.rectangle([0, 320, 640, 480], fill=(65, 70, 75))
    # Road markings
    for rx in range(40, 640, 120):
        draw.rectangle([rx, 410, rx+60, 420], fill=(240, 240, 240))
    # Red sports vintage car
    draw.rectangle([130, 250, 510, 350], fill=(215, 30, 35))
    draw.polygon([(200, 250), (250, 190), (410, 190), (460, 250)], fill=(200, 225, 245), outline=(180, 25, 30), width=3)
    # Wheels
    draw.ellipse([180, 320, 260, 400], fill=(30, 30, 30), outline=(190, 190, 190), width=6)
    draw.ellipse([380, 320, 460, 400], fill=(30, 30, 30), outline=(190, 190, 190), width=6)
    draw.rectangle([20, 20, 220, 55], fill=(30, 30, 30))
    draw.text((30, 28), "Red Vintage Coupe Car", fill=(255, 255, 255))
    img_car.save(os.path.join(output_dir, "vintage_car.jpg"), quality=95)

    # 4. Laptop Desk (Workspace Scene)
    img_desk = Image.new("RGB", (640, 480), color=(220, 195, 165)) # Wood desk
    draw = ImageDraw.Draw(img_desk)
    # Desk wood lines
    for y in range(0, 480, 50):
        draw.line([(0, y), (640, y)], fill=(195, 170, 140), width=1)
    # Laptop base
    draw.rectangle([170, 180, 470, 360], fill=(210, 215, 220), outline=(160, 165, 170), width=2)
    # Laptop screen
    draw.rectangle([190, 70, 450, 240], fill=(35, 45, 60), outline=(180, 185, 190), width=3)
    # Screen code lines
    draw.rectangle([205, 90, 320, 100], fill=(80, 210, 130))
    draw.rectangle([205, 110, 380, 120], fill=(220, 170, 90))
    draw.rectangle([205, 130, 290, 140], fill=(120, 180, 250))
    # Keyboard
    draw.rectangle([200, 255, 440, 325], fill=(70, 75, 80))
    # Coffee mug
    draw.ellipse([80, 240, 140, 300], fill=(245, 245, 245), outline=(200, 200, 200), width=2)
    draw.ellipse([90, 250, 130, 290], fill=(100, 60, 35)) # coffee liquid
    # Notebook
    draw.rectangle([490, 210, 600, 360], fill=(240, 235, 210), outline=(170, 165, 140), width=2)
    draw.rectangle([20, 20, 200, 55], fill=(30, 30, 30))
    draw.text((30, 28), "Modern Laptop Desk", fill=(255, 255, 255))
    img_desk.save(os.path.join(output_dir, "laptop_desk.jpg"), quality=95)

    # 5. Mountain Lake (Nature Scene)
    img_nature = Image.new("RGB", (640, 480), color=(160, 215, 250))
    draw = ImageDraw.Draw(img_nature)
    # Sun
    draw.ellipse([80, 40, 140, 100], fill=(255, 230, 90))
    # Mountain peaks
    draw.polygon([(40, 260), (190, 80), (340, 260)], fill=(120, 130, 145))
    draw.polygon([(190, 80), (160, 130), (220, 130)], fill=(245, 245, 250)) # snow peak
    draw.polygon([(240, 260), (410, 70), (580, 260)], fill=(100, 110, 125))
    draw.polygon([(410, 70), (370, 125), (450, 125)], fill=(245, 245, 250)) # snow peak
    # Pine forest
    for fx in range(0, 640, 30):
        draw.polygon([(fx, 280), (fx+15, 220), (fx+30, 280)], fill=(30, 90, 40))
    # Water lake
    draw.rectangle([0, 280, 640, 480], fill=(45, 125, 185))
    # Water ripples
    for wy in range(300, 480, 25):
        draw.line([(50, wy), (150, wy)], fill=(120, 180, 225), width=2)
        draw.line([(320, wy+10), (480, wy+10)], fill=(120, 180, 225), width=2)
    draw.rectangle([20, 20, 230, 55], fill=(30, 30, 30))
    draw.text((30, 28), "Alpine Mountain & Lake", fill=(255, 255, 255))
    img_nature.save(os.path.join(output_dir, "mountain_lake.jpg"), quality=95)

    print("Generated 5 sample images successfully in", output_dir)

if __name__ == "__main__":
    generate_sample_images()
