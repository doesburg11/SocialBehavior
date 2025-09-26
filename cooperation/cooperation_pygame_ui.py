import pygame
import numpy as np
import matplotlib.pyplot as plt
from cooperation_model import (
    CooperationModel, 
    cooperative_probability, 
    initial_cows, 
    high_growth_chance,     
    low_growth_chance,
    reproduction_cost,
    reproduction_threshold,
    stride_length,
    metabolism,
    grass_energy,
    max_grass_height,
    low_high_threshold
)

# --- UI Constants ---
WIDTH, HEIGHT = 600, 600
PLOT_HEIGHT = 300
SIDE_PANEL_WIDTH = 380
WINDOW_HEIGHT = HEIGHT + PLOT_HEIGHT + 80
WINDOW_WIDTH = WIDTH + SIDE_PANEL_WIDTH
GRID_SIZE = 50
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 204, 0)
RED = (255, 0, 0)
SKY = (135, 206, 235)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

class Slider:
    def __init__(self, x, y, w, label, minval, maxval, value, font, param_name, step=1.0):
        self.x = x
        self.y = y
        self.w = w
        self.rect = pygame.Rect(x, y, w, 36)
        self.label = label
        self.minval = minval
        self.maxval = maxval
        self.value = value
        self.font = font
        self.param_name = param_name
        self.step = step
        self.dragging = False
        self.editing = False
        self.text = f"{self.value:.2f}"
        self.value_rect = None

    def draw(self, screen):
        label_surf = self.font.render(f'{self.label}', True, (0, 0, 0))
        label_y = self.rect.y
        screen.blit(label_surf, (self.rect.x, label_y))
        bar_y = label_y + label_surf.get_height() + 12
        bar_height = 14
        bar_rect = pygame.Rect(self.rect.x, bar_y, self.rect.w, bar_height)
        pygame.draw.rect(screen, (180, 180, 255), bar_rect)
        pos = int((self.value-self.minval)/(self.maxval-self.minval)*self.rect.w)
        knob_w = 22
        knob_h = 34
        handle_rect = pygame.Rect(self.rect.x+pos-knob_w//2, bar_y+bar_height//2-knob_h//2, knob_w, knob_h)
        pygame.draw.rect(screen, (80, 80, 200), handle_rect)
        val_x = self.rect.x + self.rect.w + 36
        val_y = bar_y + bar_height // 2
        val_w = 70
        val_h = 28
        self.value_rect = pygame.Rect(val_x, val_y - val_h // 2, val_w, val_h)
        pygame.draw.rect(screen, (255, 255, 255), self.value_rect, border_radius=4)
        pygame.draw.rect(screen, (120, 120, 120), self.value_rect, 2, border_radius=4)
        if self.editing:
            val_surf = self.font.render(self.text, True, (0, 0, 255))
        else:
            val_surf = self.font.render(f'{self.value:.2f}', True, (0, 0, 0))
        screen.blit(val_surf, (self.value_rect.x + 6, self.value_rect.y + (self.value_rect.height - val_surf.get_height()) // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
            elif self.value_rect and self.value_rect.collidepoint(event.pos):
                self.editing = True
                self.text = f"{self.value:.2f}"
                return True
            else:
                self.editing = False
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            relx = min(max(event.pos[0] - self.rect.x, 0), self.rect.w)
            raw_value = self.minval + (self.maxval-self.minval)*relx/self.rect.w
            # Snap to step size
            steps = round((raw_value - self.minval) / self.step)
            self.value = min(max(self.minval + steps * self.step, self.minval), self.maxval)
            self.text = f"{self.value:.2f}"
            return True
        if self.editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        v = float(self.text)
                        v = min(max(v, self.minval), self.maxval)
                        # Snap to step size
                        steps = round((v - self.minval) / self.step)
                        self.value = self.minval + steps * self.step
                    except ValueError:
                        pass
                    self.editing = False
                    return True
                elif event.key == pygame.K_ESCAPE:
                    self.editing = False
                    return True
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    if len(self.text) < 8:
                        if event.unicode.isdigit() or event.unicode in '.-':
                            self.text += event.unicode
                return True
        return False

class ModelUI:
    def __init__(self):
        self.model = CooperationModel()
        self.history = []
        self.running = False
        self.ticks = 0

    def reset(self):
        self.model = CooperationModel()
        self.history = []
        self.ticks = 0
        self.running = False

    def step(self):
        self.model.step()
        self.history.append((sum(1 for cow in self.model.cows if cow.breed == 'cooperative'),
                             sum(1 for cow in self.model.cows if cow.breed == 'greedy')))
        self.ticks += 1

    def run(self, steps=1):
        for _ in range(steps):
            self.step()

def draw_grid(screen, model):
    grass = model.grass_patch.grass
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            g = grass[x, y]
            color = (0, min(255, 20 * g), 0)
            pygame.draw.rect(screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for cow in model.cows:
        cx, cy = cow.x, cow.y
        color = RED if cow.breed == 'cooperative' else SKY
        # Use floating-point positions for smooth movement
        screen_x = int(cx * CELL_SIZE + CELL_SIZE // 2)
        screen_y = int(cy * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, color, (screen_x, screen_y), CELL_SIZE // 2)
    for x in range(1, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x*CELL_SIZE, 0), (x*CELL_SIZE, HEIGHT))
    for y in range(1, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y*CELL_SIZE), (WIDTH, y*CELL_SIZE))

def draw_text(screen, text, pos, font, color=BLACK):
    surf = font.render(text, True, color)
    screen.blit(surf, pos)

def plot_history(history):
    plt.clf()
    plt.figure(figsize=(6, 2.5))
    if history:
        arr = np.array(history)
        plt.plot(arr[:, 0], label='cooperative', color='red')
        plt.plot(arr[:, 1], label='greedy', color='skyblue')
        plt.legend()
    else:
        plt.plot([], [], label='cooperative', color='red')
        plt.plot([], [], label='greedy', color='skyblue')
        plt.legend()
    plt.xlabel('time')
    plt.ylabel('count')
    plt.title('Cow Populations')
    plt.tight_layout()
    plt.savefig('coop_plot.png')
    plt.close()
    img = pygame.image.load('coop_plot.png')
    max_width = WIDTH
    max_height = PLOT_HEIGHT
    if img.get_width() > max_width or img.get_height() > max_height:
        img = pygame.transform.smoothscale(img, (max_width, max_height))
    return img

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Cooperation Model (Pygame)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    model_ui = ModelUI()
    plot_visible = [False]
    plot_img_holder = [None]
    slider_gap = 60
    slider_x = WIDTH + 30  # Increased from 10 to 30 for more space
    slider_y = 10
    sliders = [
        Slider(slider_x, slider_y + i * slider_gap, 220, label, minv, maxv, val, font, pname, step)
        for i, (label, minv, maxv, val, pname, step) in enumerate([
            ('initial-cows', 0, 100, initial_cows, 'initial_cows', 1),
            ('cooperative-probability', 0.0, 1.0, cooperative_probability, 'cooperative_probability', 0.01),
            ('stride-length', 0, 0.3, stride_length, 'stride_length', 0.01),
            ('metabolism', 0, 99, metabolism, 'metabolism', 1),
            ('reproduction-threshold', 0, 200, reproduction_threshold, 'reproduction_threshold', 1),
            ('reproduction-cost', 0, 99, reproduction_cost, 'reproduction_cost', 1),
            ('high-growth-chance', 0, 99, high_growth_chance, 'high_growth_chance', 1),
            ('low-growth-chance', 0, 99, low_growth_chance, 'low_growth_chance', 1),
            ('grass-energy', 0, 200, grass_energy, 'grass_energy', 1),
            ('max-grass-height', 1, 40, max_grass_height, 'max_grass_height', 1),
            ('low-high-threshold', 0, 99, low_high_threshold, 'low_high_threshold', 1),
        ])
    ]
    # Place buttons below the last slider
    last_slider_y = slider_y + (len(sliders)) * slider_gap
    button_rect = pygame.Rect(slider_x, last_slider_y, 100, 36)
    reset_rect = pygame.Rect(slider_x + 110, last_slider_y, 100, 36)
    plot_toggle_rect = pygame.Rect(slider_x, last_slider_y + 50, 100, 36)
    step_button_rect = pygame.Rect(slider_x + 110, last_slider_y + 50, 100, 36)
    def update_plot():
        # Only update the plot image if the plot is visible
        if plot_visible[0]:
            plot_img_holder[0] = plot_history(model_ui.history)
        else:
            plot_img_holder[0] = None
    def reset_and_update_plot():
        model_ui.reset()
        plot_visible[0] = False
        plot_img_holder[0] = None
    reset_and_update_plot()
    running = True
    while running:
        # Live-update all model parameters from sliders
        for slider in sliders:
            value = slider.value
            param = slider.param_name
            # Update globals for new model resets
            globals()[param] = value
            # Update running model parameters
            if hasattr(model_ui.model, param):
                setattr(model_ui.model, param, value)
            # Update parameters in model's grass_patch if relevant
            if hasattr(model_ui.model.grass_patch, param):
                setattr(model_ui.model.grass_patch, param, value)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            editing_slider = next((s for s in sliders if s.editing), None)
            if editing_slider and event.type == pygame.KEYDOWN:
                if editing_slider.handle_event(event):
                    continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_and_update_plot()
            slider_event_handled = False
            for slider in sliders:
                if slider.handle_event(event):
                    slider_event_handled = True
                    break
            if slider_event_handled:
                continue
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    model_ui.running = not model_ui.running
                elif reset_rect.collidepoint(event.pos):
                    reset_and_update_plot()
                elif plot_toggle_rect.collidepoint(event.pos):
                    plot_visible[0] = not plot_visible[0]
                    if plot_visible[0]:
                        update_plot()
                    else:
                        plot_img_holder[0] = None
                elif step_button_rect.collidepoint(event.pos) and not model_ui.running:
                    model_ui.step()
                    if plot_visible[0]:
                        update_plot()
        if model_ui.running:
            model_ui.step()
            # Only update plot if visible
            if plot_visible[0]:
                update_plot()
        screen.fill(WHITE)
        draw_grid(screen, model_ui.model)
        for slider in sliders:
            slider.draw(screen)
        pygame.draw.rect(screen, (100, 200, 100) if model_ui.running else (200, 100, 100), button_rect, border_radius=8)
        btn_label = 'Stop' if model_ui.running else 'Start'
        btn_surf = font.render(btn_label, True, (255, 255, 255))
        screen.blit(btn_surf, (button_rect.x+18, button_rect.y+8))
        pygame.draw.rect(screen, (100, 100, 200), reset_rect, border_radius=8)
        reset_surf = font.render('Reset', True, (255, 255, 255))
        screen.blit(reset_surf, (reset_rect.x+18, reset_rect.y+8))
        pygame.draw.rect(screen, (120, 120, 220) if plot_visible[0] else (180, 180, 180), plot_toggle_rect, border_radius=8)
        plot_label = 'Plot: ON' if plot_visible[0] else 'Plot: OFF'
        plot_surf = font.render(plot_label, True, (255, 255, 255))
        screen.blit(plot_surf, (plot_toggle_rect.x+10, plot_toggle_rect.y+8))
        step_color = (100, 200, 100) if not model_ui.running else (180, 180, 180)
        pygame.draw.rect(screen, step_color, step_button_rect, border_radius=8)
        step_surf = font.render('Step', True, (255, 255, 255))
        screen.blit(step_surf, (step_button_rect.x+25, step_button_rect.y+8))
        # Place stats below the last button
        stats_y = step_button_rect.y + 50
        draw_text(screen, f'Ticks: {model_ui.ticks}', (slider_x, stats_y), font)
        coop = sum(1 for cow in model_ui.model.cows if cow.breed == 'cooperative')
        greedy = sum(1 for cow in model_ui.model.cows if cow.breed == 'greedy')
        draw_text(screen, f'Cooperative: {coop}', (slider_x, stats_y + 40), font, RED)
        draw_text(screen, f'Greedy: {greedy}', (slider_x, stats_y + 70), font, SKY)
        if plot_visible[0]:
            plot_img = plot_img_holder[0]
            if plot_img:
                plot_x = (WIDTH - plot_img.get_width()) // 2
                plot_y = HEIGHT + 30
                screen.blit(plot_img, (plot_x, plot_y))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == '__main__':
    main()
