"""
main.py -- the Courier game: menu, play mode, watch-the-AI mode.

    python3 main.py

SCENES:  menu -> level select -> game
MODES:   "play"  = YOU control the robot (feel how hard it is for the AI)
         "watch" = the AI controls the robot
"""
import math
import random
import sys
import pygame
from courier_env import CourierEnv, UP, DOWN, LEFT, RIGHT
from levels import LEVELS, RANDOM_BASELINE
from agents import get_agent
from renderer import (W, H, CELL, BX, BY, Renderer, background, text, panel, draw_button,
                      draw_robot, alpha_circle)

# keyboard -> action (arrows and WASD both work)
KEYS = {pygame.K_UP: UP, pygame.K_w: UP, pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
        pygame.K_LEFT: LEFT, pygame.K_a: LEFT, pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT}
# keys 1-4 = speed of the AI, as seconds per move
SPEEDS = {pygame.K_1: 0.5, pygame.K_2: 0.25, pygame.K_3: 0.12, pygame.K_4: 0.05}


class App:
    def __init__(self, screen):
        self.screen = screen
        self.bg = background()
        self.scene = "menu"      # "menu", "levels" or "game"
        self.mode = None         # "play" or "watch"
        self.t = 0.0             # time since start, drives animations
        self.mouse = (0, 0)
        self.note = ""           # small message, e.g. "Level 2 is coming soon"
        # menu buttons
        self.menu_btns = [("play", pygame.Rect(300, 325, 360, 68), "PLAY IT YOURSELF", "Feel what the AI feels"),
                          ("watch", pygame.Rect(300, 407, 360, 68), "WATCH THE AI", "Pick a level and see it play"),
                          ("quit", pygame.Rect(300, 489, 360, 68), "QUIT", None)]
        # level cards: 4 per row, 2 rows
        self.cards = [pygame.Rect(40 + (i % 4) * 225, 190 + (i // 4) * 180, 205, 160) for i in range(len(LEVELS))]

    # ------------------------------------------------------------ game setup
    def start_game(self, level):
        self.level = level
        self.env = CourierEnv(level=level)
        self.seed = random.randint(0, 10**6)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.view = Renderer(self.env)
        self.agent = get_agent(level, self.env)
        self.over, self.info = False, {}
        self.timer, self.wait, self.paused = 0.0, 0.0, False
        self.interval = 0.25
        self.games, self.delivered, self.scores = 0, 0, []     # watch-mode statistics
        self.scene = "game"

    def new_game(self):
        self.seed += 1
        self.obs, _ = self.env.reset(seed=self.seed)
        self.view.reset_view()
        self.over, self.info, self.wait = False, {}, 0.0

    def do_step(self, action):
        """Make one move and tell the graphics what happened."""
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self.view.on_step(action, self.info)
        self.over = term or trunc
        if self.over and self.mode == "watch":                 # record AI statistics
            self.games += 1
            self.delivered += self.info["delivered"]
            self.scores.append(self.info["total_reward"])
            self.wait = 1.4                                    # pause, then auto-restart

    # ------------------------------------------------------------ input
    def handle(self, e):
        if e.type == pygame.QUIT:
            sys.exit()
        if e.type == pygame.MOUSEMOTION:
            self.mouse = e.pos
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.click(e.pos)
        if e.type == pygame.KEYDOWN:
            self.key(e.key)

    def click(self, pos):
        if self.scene == "menu":
            for name, rect, _, _ in self.menu_btns:
                if rect.collidepoint(pos):
                    if name == "quit":
                        sys.exit()
                    self.mode, self.scene, self.note = name, "levels", ""
        elif self.scene == "levels":
            for lv, rect in zip(LEVELS, self.cards):
                if rect.collidepoint(pos):
                    if lv["ready"]:
                        self.start_game(lv["id"])
                    else:
                        self.note = f"Level {lv['id']} is coming soon - new levels unlock every week!"

    def key(self, k):
        if k == pygame.K_ESCAPE:
            if self.scene == "menu":
                sys.exit()
            self.scene = "levels" if self.scene == "game" else "menu"
            self.note = ""
        elif self.scene == "game":
            if k == pygame.K_r:
                self.new_game()
                self.games = self.delivered = 0
                self.scores = []
            elif self.mode == "play" and not self.over and k in KEYS:
                self.do_step(KEYS[k])
            elif self.mode == "watch":
                if k == pygame.K_SPACE:
                    self.paused = not self.paused
                elif k in SPEEDS:
                    self.interval = SPEEDS[k]

    # ------------------------------------------------------------ update
    def update(self, dt):
        self.t += dt
        if self.scene != "game":
            return
        self.view.update(dt)
        if self.mode == "watch" and not self.paused:
            if self.over:                                      # short pause between games
                self.wait -= dt
                if self.wait <= 0:
                    self.new_game()
            else:
                self.timer += dt
                if self.timer >= self.interval:
                    self.timer = 0
                    self.do_step(self.agent.act(self.obs))

    # ------------------------------------------------------------ drawing
    def draw(self):
        s = self.screen
        s.blit(self.bg, (0, 0))
        {"menu": self.draw_menu, "levels": self.draw_levels, "game": self.draw_game}[self.scene](s)

    def draw_menu(self, s):
        # mascot: draw the robot small, then scale it up 2x
        small = pygame.Surface((120, 130), pygame.SRCALPHA)
        draw_robot(small, 60, 70, self.t, (1, 0), False)
        big = pygame.transform.smoothscale(small, (216, 234))
        s.blit(big, (480 - 108, 22 + int(math.sin(self.t * 2) * 5)))
        text(s, "COURIER", (480, 255), 64, center=True)
        text(s, "Can an AI learn to deliver a package?", (480, 298), 20, (148, 163, 184), True, False)
        for _, rect, label, sub in self.menu_btns:
            draw_button(s, rect, label, sub, rect.collidepoint(self.mouse))
        text(s, "Week 1  -  Level 1 is ready, more levels unlock every week", (480, 605), 16, (100, 116, 139), True, False)

    def draw_levels(self, s):
        title = "WATCH THE AI" if self.mode == "watch" else "PLAY IT YOURSELF"
        text(s, title, (480, 70), 44, center=True)
        text(s, "Choose a level", (480, 118), 22, (148, 163, 184), True, False)
        for lv, rect in zip(LEVELS, self.cards):
            hover = rect.collidepoint(self.mouse)
            ready = lv["ready"]
            panel(s, rect, (14, 116, 144) if (hover and ready) else (30, 41, 59), 240)
            pygame.draw.rect(s, (56, 189, 248) if ready else (51, 65, 85), rect, 2, border_radius=16)
            col = (255, 255, 255) if ready else (100, 116, 139)
            text(s, str(lv["id"]), (rect.x + 20, rect.y + 12), 46, (250, 204, 21) if ready else (71, 85, 105))
            text(s, lv["name"], (rect.x + 20, rect.y + 72), 22, col)
            text(s, lv["desc"], (rect.x + 20, rect.y + 100), 15, (148, 163, 184) if ready else (71, 85, 105), False, False)
            if ready:
                text(s, "READY", (rect.x + 20, rect.y + 128), 16, (74, 222, 128))
            else:                                              # padlock icon + label
                lx, ly = rect.right - 40, rect.y + 20
                pygame.draw.arc(s, (100, 116, 139), (lx + 3, ly - 6, 16, 18), 0, 3.14, 3)
                pygame.draw.rect(s, (100, 116, 139), (lx, ly + 4, 22, 16), border_radius=3)
                text(s, "COMING SOON", (rect.x + 20, rect.y + 128), 15, (100, 116, 139))
        if self.note:
            text(s, self.note, (480, 575), 20, (251, 191, 36), True)
        text(s, "ESC = back", (480, 610), 15, (100, 116, 139), True, False)

    def draw_game(self, s):
        env = self.env
        lv = LEVELS[self.level - 1]
        stage = self.bg.copy()                                 # draw on a copy so the map can shake
        self.view.draw(stage)
        s.blit(stage, self.view.offset())
        text(s, f"COURIER  -  Level {lv['id']}: {lv['name']}", (40, 26), 28)
        # ---- right-hand info panel
        p = pygame.Rect(610, BY - 14, 310, env.size * CELL + 28)
        panel(s, p, (15, 23, 42), 225, 20)
        x = p.x + 22
        head = "YOU ARE THE ROBOT" if self.mode == "play" else "AI IS PLAYING"
        text(s, head, (x, p.y + 20), 20, (56, 189, 248))
        text(s, "SCORE", (x, p.y + 62), 14, (148, 163, 184))
        text(s, f"{env.total_reward:.0f}", (x, p.y + 80), 54)
        text(s, f"MOVES  {env.steps} / {env.max_steps}", (x, p.y + 150), 14, (148, 163, 184))
        bar = pygame.Rect(x, p.y + 172, 266, 12)
        pygame.draw.rect(s, (51, 65, 85), bar, border_radius=6)
        full = env.steps / env.max_steps
        pygame.draw.rect(s, (251, 191, 36) if full > 0.75 else (56, 189, 248),
                         (bar.x, bar.y, int(bar.w * full), bar.h), border_radius=6)
        status = "Carrying the package!" if env.carrying else "Go find the package"
        text(s, status, (x, p.y + 200), 18, (74, 222, 128) if env.carrying else (226, 232, 240))
        y = p.y + 245
        if self.mode == "play":
            lines = ["Arrow keys or WASD to move", "Walk to the box, then to GOAL.",
                     "Each move costs 1 point.", "Hitting the edge costs 5.",
                     f"A random robot averages {RANDOM_BASELINE}.", "", "R = restart     ESC = back"]
        else:
            avg = sum(self.scores) / len(self.scores) if self.scores else 0
            lines = [self.agent.name, f"Games played: {self.games}",
                     f"Deliveries: {self.delivered}", f"Average score: {avg:.0f}", "",
                     "SPACE pause   1-4 speed", "R reset stats   ESC back"]
            if self.paused:
                text(s, "PAUSED", (p.right - 100, p.y + 20), 20, (251, 191, 36))
        for line in lines:
            text(s, line, (x, y), 16, (203, 213, 225), False, False)
            y += 26
        # ---- end-of-game banner on the map
        if self.over:
            ok = self.info["delivered"]
            box = pygame.Rect(0, 0, 380, 150)
            box.center = (BX + env.size * CELL // 2, BY + env.size * CELL // 2)
            panel(s, box, (15, 23, 42), 235, 20)
            pygame.draw.rect(s, (74, 222, 128) if ok else (248, 113, 113), box, 3, border_radius=20)
            text(s, "DELIVERED!" if ok else "OUT OF MOVES", (box.centerx, box.y + 42), 40, center=True,
                 color=(74, 222, 128) if ok else (248, 113, 113))
            text(s, f"Score {self.info['total_reward']:.0f}   in {self.info['steps']} moves", (box.centerx, box.y + 88), 22, center=True)
            hint = "Press R to play again" if self.mode == "play" else "Next game starting..."
            text(s, hint, (box.centerx, box.y + 120), 16, (148, 163, 184), True, False)


def main():
    pygame.init()
    pygame.display.set_caption("Courier")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    app = App(screen)
    while True:
        dt = clock.tick(60) / 1000          # seconds since the last frame
        for e in pygame.event.get():
            app.handle(e)
        app.update(dt)
        app.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
