import pygame
import numpy as np
import jetson.inference
import jetson.utils

pygame.init()

# ====== MODEL SETUP ======
net = jetson.inference.detectNet(
    model="/home/nvidia11/jetson-inference/python/training/detection/ssd/models/test_detect_model/ssd-mobilenet.onnx",
    labels="/home/nvidia11/jetson-inference/python/training/detection/ssd/models/test_detect_model/labels.txt",
    input_blob="input_0",
    output_cvg="scores",
    output_bbox="boxes",
    threshold=0.75
)

# ====== DISPLAY SETUP ======
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("NVIDIA FINAL PROJECT")
clock = pygame.time.Clock()

# ====== CAMERA ======
cam = jetson.utils.videoSource("/dev/video0")
show_webcam = True

# ====== ASSETS ======
background = pygame.image.load("assets/background.png")
background = pygame.transform.scale(background, (screen_width, screen_height))
game_background = pygame.image.load("assets/game.background.png")
game_background = pygame.transform.scale(game_background, (screen_width, screen_height))

left_mage = pygame.image.load("assets/left.mage.sprite.png").convert_alpha()
resized_left_mage = pygame.transform.scale_by(left_mage, 0.25)
right_mage = pygame.image.load("assets/right.mage.sprite.png").convert_alpha()
resized_right_mage = pygame.transform.scale_by(right_mage, 0.25)

webcam_border = pygame.image.load("assets/webcam_frame.png").convert_alpha()

# ====== BUTTONS ======
button_width = 260
button_height = 70
play_button_x = (screen_width - button_width) // 2
play_button_y = int(screen_height * 0.75)
tutorial_button_x = play_button_x
tutorial_button_y = play_button_y + button_height + 20

play_button = pygame.Rect(play_button_x, play_button_y, button_width, button_height)
tutorial_button = pygame.Rect(tutorial_button_x, tutorial_button_y, button_width, button_height)

# ====== GAME STATES ======
MENU = "menu"
GAME = "game"
TUTORIAL = "tutorial"
current_screen = MENU

# ====== CAMERA FRAME ======
frame_surface = None
cam_size = (220, 160)

# ====== SPELL SETTINGS ======
SPELL_DURATION = 1.0
SPELL_COOLDOWN = 2.0
SPELL_SCALE = 0.5

# Per-spell positions (manual)
SPELL_POSITIONS = {
    "thunder": (1150, 300),  # above head
    "fire":    (1100, 300),  # on top of mage
    "ice":     (1050, 350),  # in front of mage
    "rock":    (1050, 450)   # in front of mage
}

spell_active = False
spell_timer = 0
cooldown_timer = 0
active_spell_name = None

# ====== SPRITESHEET LOADING ======
def load_spritesheet(filename, frame_width, frame_height):
    sheet = pygame.image.load(filename).convert_alpha()
    sheet_width, sheet_height = sheet.get_size()
    frames = []
    for y in range(0, sheet_height, frame_height):
        for x in range(0, sheet_width, frame_width):
            frame = sheet.subsurface(pygame.Rect(x, y, frame_width, frame_height))
            frames.append(frame)
    return frames

# Load each spell's spritesheet (800x800 frames)
spell_animations = {
    "rock": load_spritesheet("assets/rock.png", 800, 800),
    "fire": load_spritesheet("assets/fireball.png", 800, 800),
    "ice": load_spritesheet("assets/ice.png", 800, 800),
    "thunder": load_spritesheet("assets/thunder.png", 800, 800)
}

# ====== MAIN LOOP ======
running = True
while running:
    dt = clock.tick(60) / 1000

    # --- EVENTS ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif current_screen == MENU and event.type == pygame.MOUSEBUTTONDOWN:
            if play_button.collidepoint(event.pos):
                current_screen = GAME
            elif tutorial_button.collidepoint(event.pos):
                current_screen = TUTORIAL
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                show_webcam = not show_webcam
            elif event.key == pygame.K_ESCAPE:
                running = False

    # --- CAPTURE & DETECT ---
    detections = []
    if show_webcam:
        frame = cam.Capture()
        if frame:
            detections = net.Detect(frame)
            frame_np = jetson.utils.cudaToNumpy(frame)
            frame_surface = pygame.surfarray.make_surface(frame_np.swapaxes(0, 1))
            frame_surface = pygame.transform.scale(frame_surface, cam_size)

    # --- DRAW WEBCAM + SPELL ---
    def draw_webcam():
        global spell_active, spell_timer, cooldown_timer, active_spell_name

        if show_webcam and frame_surface:
            cam_x = (screen_width - cam_size[0]) // 2
            cam_y = 60
            screen.blit(frame_surface, (cam_x, cam_y))

            for det in detections:
                label = net.GetClassDesc(det.ClassID)
                if (current_screen == GAME and label in spell_animations
                    and not spell_active and pygame.time.get_ticks() - cooldown_timer > SPELL_COOLDOWN * 1000):
                    spell_active = True
                    spell_timer = pygame.time.get_ticks()
                    active_spell_name = label

            if spell_active and current_screen == GAME:
                elapsed = (pygame.time.get_ticks() - spell_timer) / 1000
                frames = spell_animations[active_spell_name]
                frame_index = int((elapsed / SPELL_DURATION) * len(frames))
                if frame_index >= len(frames):
                    frame_index = len(frames) - 1

                spell_frame = frames[frame_index]
                w = int(spell_frame.get_width() * SPELL_SCALE)
                h = int(spell_frame.get_height() * SPELL_SCALE)
                spell_frame_scaled = pygame.transform.scale(spell_frame, (w, h))

                pos_x, pos_y = SPELL_POSITIONS[active_spell_name]
                screen.blit(spell_frame_scaled, (pos_x, pos_y))

                if elapsed > SPELL_DURATION:
                    spell_active = False
                    cooldown_timer = pygame.time.get_ticks()

            border_width = 380
            border_height = 380
            scaled_border = pygame.transform.scale(webcam_border, (border_width, border_height))
            border_x = 610
            border_y = -50
            screen.blit(scaled_border, (border_x, border_y))

    # --- DRAW SCREEN ---
    if current_screen == MENU:
        screen.blit(background, (0, 0))
        draw_webcam()

    elif current_screen == GAME:
        screen.blit(game_background, (0, 0))
        mage_left_x = 150
        mage_right_x = 1200
        mage_y = 300
        screen.blit(resized_left_mage, (mage_left_x, mage_y))
        screen.blit(resized_right_mage, (mage_right_x, mage_y))
        draw_webcam()

    elif current_screen == TUTORIAL:
        screen.fill((0, 0, 0))
        mage_right_x = 650
        mage_y = 250
        screen.blit(resized_right_mage, (mage_right_x, mage_y))
        draw_webcam()

    pygame.display.flip()

pygame.quit()