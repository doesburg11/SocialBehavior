

# --- ADVANCED UI RESTORE (NO STEP BUTTON) ---
import pygame
import numpy as np
import matplotlib.pyplot as plt
from altruism_model import AltruismModel, Params

# --- UI Constants ---
WIDTH, HEIGHT = 600, 600
PLOT_HEIGHT = 300
SIDE_PANEL_WIDTH = 380
WINDOW_HEIGHT = HEIGHT + PLOT_HEIGHT + 80
WINDOW_WIDTH = WIDTH + SIDE_PANEL_WIDTH
GRID_SIZE = 51
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 204, 0)
PINK = (255, 102, 179)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


class Slider:
    def __init__(self, x, y, w, label, minval, maxval, value, font, param_name, scale=1.0):
        self.x = x
        self.y = y
        self.w = w
        self.scale = scale
        self.rect = pygame.Rect(x, y, w, int(36 * scale))
        self.label = label
        self.minval = minval
        self.maxval = maxval
        self.value = value
        self.font = font
        self.param_name = param_name
        self.dragging = False
        # For editable value
        self.editing = False
        self.text = f"{self.value:.2f}"
        self.value_rect = None

    def draw(self, screen):
        # Draw label above the slider
        label_surf = self.font.render(f'{self.label}', True, (0, 0, 0))
        label_y = self.rect.y
        screen.blit(label_surf, (self.rect.x, label_y))
        # Draw bar below label
        bar_y = label_y + label_surf.get_height() + int(12 * self.scale)  # reduced space between label and bar
        bar_height = max(int(14 * self.scale), 8)
        bar_rect = pygame.Rect(self.rect.x, bar_y, self.rect.w, bar_height)
        pygame.draw.rect(screen, (180, 180, 255), bar_rect)
        # Draw handle (knob)
        pos = int((self.value-self.minval)/(self.maxval-self.minval)*self.rect.w)
        knob_w = max(int(22 * self.scale), 14)
        knob_h = max(int(34 * self.scale), 20)
        handle_rect = pygame.Rect(self.rect.x+pos-knob_w//2, bar_y+bar_height//2-knob_h//2, knob_w, knob_h)
        pygame.draw.rect(screen, (80, 80, 200), handle_rect)
        # Draw value to the right of the slider (as editable box)
        val_x = self.rect.x + self.rect.w + int(36 * self.scale)
        val_y = bar_y + bar_height // 2
        val_w = int(70 * self.scale)
        val_h = int(28 * self.scale)
        self.value_rect = pygame.Rect(val_x, val_y - val_h // 2, val_w, val_h)
        pygame.draw.rect(screen, (255, 255, 255), self.value_rect, border_radius=4)
        pygame.draw.rect(screen, (120, 120, 120), self.value_rect, 2, border_radius=4)
        if self.editing:
            # Draw text input
            val_surf = self.font.render(self.text, True, (0, 0, 255))
        else:
            val_surf = self.font.render(f'{self.value:.2f}', True, (0, 0, 0))
        screen.blit(
            val_surf,
            (
                self.value_rect.x + 6,
                self.value_rect.y + (self.value_rect.height - val_surf.get_height()) // 2
            )
        )

    def handle_event(self, event):
        # Handle slider drag
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
            self.value = self.minval + (self.maxval-self.minval)*relx/self.rect.w
            self.value = min(max(self.value, self.minval), self.maxval)
            self.text = f"{self.value:.2f}"
            return True
        # Handle text input for value
        if self.editing:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        v = float(self.text)
                        v = min(max(v, self.minval), self.maxval)
                        self.value = v
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
        self.params = Params(width=GRID_SIZE, height=GRID_SIZE)
        self.model = AltruismModel(self.params)
        self.history = []
        self.running = False
        self.ticks = 0

    def reset(self):
        self.model = AltruismModel(self.params)
        self.history = []
        self.ticks = 0
        self.running = False

    def step(self):
        self.model.go()
        self.history.append(self.model.counts())
        self.ticks += 1

    def run(self, steps=1):
        for _ in range(steps):
            self.step()


def draw_grid(screen, model):
    for y in range(model.p.height):
        for x in range(model.p.width):
            val = model.pcolor[y, x]
            color = BLACK if val == 0 else GREEN if val == 1 else PINK
            pygame.draw.rect(screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for x in range(1, model.p.width):
        pygame.draw.line(screen, GRAY, (x*CELL_SIZE, 0), (x*CELL_SIZE, model.p.height*CELL_SIZE))
    for y in range(1, model.p.height):
        pygame.draw.line(screen, GRAY, (0, y*CELL_SIZE), (model.p.width*CELL_SIZE, y*CELL_SIZE))


def draw_text(screen, text, pos, font, color=BLACK):
    surf = font.render(text, True, color)
    screen.blit(surf, pos)


def plot_history(history):
    plt.clf()
    plt.figure(figsize=(6, 2.5))
    if history:
        arr = np.array(history)
        plt.plot(arr[:, 0], label='altruists', color='magenta')
        plt.plot(arr[:, 1], label='selfish', color='green')
        plt.legend()
    else:
        plt.plot([], [], label='altruists', color='magenta')
        plt.plot([], [], label='selfish', color='green')
        plt.legend()
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.title('Populations')
    plt.tight_layout()
    plt.savefig('pop_plot.png')
    plt.close()
    img = pygame.image.load('pop_plot.png')
    max_width = WIDTH
    max_height = PLOT_HEIGHT
    if img.get_width() > max_width or img.get_height() > max_height:
        img = pygame.transform.smoothscale(img, (max_width, max_height))
    return img


def main():
    # Define zoom before get_rects so it is in scope
    zoom = [1.0]

    def get_rects():
        scale = zoom[0]
        bx = WIDTH + 10
        slider_gap = int(56 * scale)
        slider_y = int(10 * scale)
        stats_height = int(120 * scale)
        by = slider_y + 6 * slider_gap + stats_height + int(10 * scale)
        bw = int(100 * scale)
        bh = int(36 * scale)
        button_rect = pygame.Rect(bx, by, bw, bh)
        reset_rect = pygame.Rect(bx + bw + 10, by, bw, bh)
        plot_toggle_rect = pygame.Rect(bx, by + int(50 * scale), bw, bh)
        step_button_rect = pygame.Rect(bx + bw + 10, by + int(50 * scale), bw, bh)
        return button_rect, reset_rect, plot_toggle_rect, step_button_rect

    # Initialize pygame and screen before recalc_layout
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Altruism Model (Pygame)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    model_ui = ModelUI()
    plot_visible = [False]
    plot_img_holder = [None]
    sliders = [[]]  # mutable container for sliders list

    # Button rects, initialized after recalc_layout
    button_rect, reset_rect, plot_toggle_rect, step_button_rect = None, None, None, None

    def recalc_layout():
        scale = zoom[0]
        global WIDTH, HEIGHT, PLOT_HEIGHT, WINDOW_HEIGHT, GRID_SIZE, CELL_SIZE, font
        WIDTH = int(600 * scale)
        HEIGHT = int(600 * scale)
        PLOT_HEIGHT = int(300 * scale)
        WINDOW_HEIGHT = HEIGHT + PLOT_HEIGHT + int(80 * scale)
        GRID_SIZE = 51
        CELL_SIZE = WIDTH // GRID_SIZE
        font = pygame.font.SysFont(None, int(24 * scale))
        slider_gap = int(56 * scale)
        slider_x = WIDTH + 10
        slider_w = int(220 * scale)
        slider_y = int(10 * scale)
        slider_defs = [
            (slider_x, slider_y + i * slider_gap, slider_w, label, minv, maxv, val, font, pname, scale)
            for i, (label, minv, maxv, val, pname) in enumerate([
                ('altruistic-probability', 0.0, 1.0, model_ui.params.altruistic_probability, 'altruistic_probability'),
                ('selfish-probability', 0.0, 1.0, model_ui.params.selfish_probability, 'selfish_probability'),
                ('cost-of-altruism', 0.0, 1.0, model_ui.params.cost_of_altruism, 'cost_of_altruism'),
                ('benefit-from-altruism', 0.0, 1.0, model_ui.params.benefit_from_altruism, 'benefit_from_altruism'),
                ('disease', 0.0, 1.0, model_ui.params.disease, 'disease'),
                ('harshness', 0.0, 1.0, model_ui.params.harshness, 'harshness'),
            ])
        ]
        sliders[0] = [Slider(*args) for args in slider_defs]
        nonlocal button_rect, reset_rect, plot_toggle_rect, step_button_rect
        button_rect, reset_rect, plot_toggle_rect, step_button_rect = get_rects()

    def update_plot(resize_window=True):
        if plot_visible[0]:
            plot_img_holder[0] = plot_history(model_ui.history)
        else:
            plot_img_holder[0] = None
        # Window size is fixed, so only recalc layout and set_mode once at startup

    def reset_and_update_plot():
        model_ui.reset()
        plot_visible[0] = False
        update_plot(resize_window=True)

    recalc_layout()
    reset_and_update_plot()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            # Always send KEYDOWN to editing slider if any
            editing_slider = next((s for s in sliders[0] if s.editing), None)
            if editing_slider and event.type == pygame.KEYDOWN:
                if editing_slider.handle_event(event):
                    if not editing_slider.editing:
                        setattr(model_ui.params, editing_slider.param_name, editing_slider.value)
                        if plot_visible[0]:
                            model_ui.reset()
                            update_plot(resize_window=False)
                        else:
                            model_ui.reset()
                    continue
            # Handle R key for reset
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_and_update_plot()
            # Handle slider events (mouse and non-editing key events)
            slider_event_handled = False
            for slider in sliders[0]:
                if slider.handle_event(event):
                    slider_event_handled = True
                    if not slider.editing:
                        setattr(model_ui.params, slider.param_name, slider.value)
                        if plot_visible[0]:
                            model_ui.reset()
                            update_plot(resize_window=False)
                        else:
                            model_ui.reset()
                    break
            if slider_event_handled:
                continue
            # Handle button events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    model_ui.running = not model_ui.running
                elif reset_rect.collidepoint(event.pos):
                    reset_and_update_plot()
                elif plot_toggle_rect.collidepoint(event.pos):
                    plot_visible[0] = not plot_visible[0]
                    update_plot(resize_window=True)
                elif step_button_rect.collidepoint(event.pos) and not model_ui.running:
                    model_ui.step()
                    if plot_visible[0]:
                        update_plot(resize_window=False)
            # Only update model and plot on slider release, not on drag
            if event.type == pygame.MOUSEBUTTONUP:
                for slider in sliders[0]:
                    if slider.dragging:
                        slider.dragging = False
                        setattr(model_ui.params, slider.param_name, slider.value)
                        if plot_visible[0]:
                            model_ui.reset()
                            update_plot(resize_window=False)
                        else:
                            model_ui.reset()

        if model_ui.running:
            model_ui.step()
            if plot_visible[0]:
                update_plot(resize_window=False)

        screen.fill(WHITE)
        draw_grid(screen, model_ui.model)
        for slider in sliders[0]:
            slider.draw(screen)
    # Zoom buttons removed
        pygame.draw.rect(screen, (100, 200, 100) if model_ui.running else (200, 100, 100), button_rect, border_radius=8)
        btn_label = 'Stop' if model_ui.running else 'Start'
        btn_surf = font.render(btn_label, True, (255, 255, 255))
        screen.blit(btn_surf, (button_rect.x+int(18*zoom[0]), button_rect.y+int(8*zoom[0])))
        pygame.draw.rect(screen, (100, 100, 200), reset_rect, border_radius=8)
        reset_surf = font.render('Reset', True, (255, 255, 255))
        screen.blit(reset_surf, (reset_rect.x+int(18*zoom[0]), reset_rect.y+int(8*zoom[0])))
        pygame.draw.rect(
            screen,
            (120, 120, 220) if plot_visible[0] else (180, 180, 180),
            plot_toggle_rect,
            border_radius=8
        )
        plot_label = 'Plot: ON' if plot_visible[0] else 'Plot: OFF'
        plot_surf = font.render(plot_label, True, (255, 255, 255))
        screen.blit(plot_surf, (plot_toggle_rect.x+int(10*zoom[0]), plot_toggle_rect.y+int(8*zoom[0])))
        # Draw Step button (active only if stopped)
        step_color = (100, 200, 100) if not model_ui.running else (180, 180, 180)
        pygame.draw.rect(screen, step_color, step_button_rect, border_radius=8)
        step_surf = font.render('Step', True, (255, 255, 255))
        screen.blit(step_surf, (step_button_rect.x+int(25*zoom[0]), step_button_rect.y+int(8*zoom[0])))
        # Draw text and stats, scaled
        draw_text(screen, f'Ticks: {model_ui.ticks}', (WIDTH+int(10*zoom[0]), int(340*zoom[0])), font)
        pink, green, black = model_ui.model.counts()
        draw_text(screen, f'Altruists: {pink}', (WIDTH+int(10*zoom[0]), int(380*zoom[0])), font, PINK)
        draw_text(screen, f'Selfish: {green}', (WIDTH+int(10*zoom[0]), int(410*zoom[0])), font, GREEN)
        draw_text(screen, f'Black: {black}', (WIDTH+int(10*zoom[0]), int(440*zoom[0])), font, BLACK)
        if plot_visible[0]:
            plot_img = plot_img_holder[0]
            if plot_img:
                plot_x = (WIDTH - plot_img.get_width()) // 2
                plot_y = HEIGHT + int(30*zoom[0])
                screen.blit(plot_img, (plot_x, plot_y))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == '__main__':
    main()
