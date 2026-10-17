"""
renderer.py -- ALL the drawing: the map, the robot, the package, the effects
and small helpers (text, buttons) used by the menus.
Nothing here decides game rules; it only draws what the game says.
"""
import math
import random
import pygame
from courier_env import MOVES

W, H = 960, 640          # window size
CELL = 104               # size of one map square
BX, BY = 40, 80          # top-left corner of the map on screen

# ---------------------------------------------------------------- helpers
_fonts = {}


def font(size, bold=True):
    """Cached font (loading fonts every frame would be slow)."""
    if (size, bold) not in _fonts:
        _fonts[(size, bold)] = pygame.font.SysFont(
            "helveticaneue,arialroundedmtbold,arial,dejavusans", size, bold=bold)
    return _fonts[(size, bold)]


def text(s, msg, pos, size=24, color=(255, 255, 255), center=False, bold=True):
    """Draw text. center=True puts the middle of the text at pos."""
    img = font(size, bold).render(str(msg), True, color)
    rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
    s.blit(img, rect)
    return rect


def alpha_circle(s, color, alpha, pos, r, width=0):
    """A semi-transparent circle (used for glows, trails, sparks)."""
    t = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(t, (*color, alpha), (r + 2, r + 2), r, width)
    s.blit(t, (pos[0] - r - 2, pos[1] - r - 2))


def panel(s, rect, color=(15, 23, 42), alpha=215, radius=16):
    """A rounded, semi-transparent box (background of info panels)."""
    t = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(t, (*color, alpha), t.get_rect(), border_radius=radius)
    s.blit(t, rect.topleft)


def background():
    """Dark blue gradient used behind everything (made once, reused)."""
    surf = pygame.Surface((W, H))
    for y in range(H):
        k = y / H
        pygame.draw.line(surf, (int(15 + 10 * k), int(23 + 14 * k), int(42 + 30 * k)), (0, y), (W, y))
    return surf


def draw_button(s, rect, label, sub=None, hover=False, accent=(56, 189, 248)):
    """A clickable button. Lights up when the mouse is over it."""
    panel(s, rect, (14, 116, 144) if hover else (30, 41, 59), 255 if hover else 235)
    pygame.draw.rect(s, accent if hover else (71, 85, 105), rect, 2, border_radius=16)
    text(s, label, (rect.centerx, rect.centery - (11 if sub else 0)), 28, center=True)
    if sub:
        text(s, sub, (rect.centerx, rect.centery + 17), 16, (148, 163, 184), True, False)


def draw_box(s, r):
    """A cardboard box with a tape stripe."""
    pygame.draw.rect(s, (146, 92, 40), r.move(0, 3), border_radius=5)
    pygame.draw.rect(s, (196, 134, 66), r, border_radius=5)
    pygame.draw.rect(s, (232, 196, 140), (r.centerx - max(2, r.w // 10), r.top, max(4, r.w // 5), r.h))
    pygame.draw.rect(s, (120, 74, 30), r, 2, border_radius=5)


def draw_robot(s, cx, cy, t, facing=(1, 0), carrying=False):
    """The robot: wheels, body, glowing eyes, blinking lights, antenna.
    facing = (row change, column change) of its last move: eyes look that way."""
    cx, cy = int(cx), int(cy)
    pygame.draw.ellipse(s, (10, 15, 28), (cx - 32, cy + 20, 64, 18))              # shadow
    for dx in (-33, 25):                                                           # 4 wheels
        for dy in (-24, 6):
            pygame.draw.rect(s, (15, 23, 42), (cx + dx, cy + dy, 8, 18), border_radius=3)
    body = pygame.Rect(cx - 27, cy - 25, 54, 52)
    pygame.draw.rect(s, (37, 99, 235), body, border_radius=13)                     # dark body
    pygame.draw.rect(s, (59, 130, 246), (body.x, body.y, body.w, 34), border_radius=13)  # light top
    pygame.draw.rect(s, (147, 197, 253), (cx - 20, cy - 23, 40, 4), border_radius=2)     # shine
    pygame.draw.rect(s, (15, 23, 42), (cx - 20, cy - 16, 40, 24), border_radius=9)       # face screen
    blink = (t % 3.2) < 0.12                                                       # eyes blink
    for ex in (-9, 9):
        x, y = cx + ex + facing[1] * 3, cy - 4 + facing[0] * 2
        alpha_circle(s, (34, 211, 238), 70, (x, y), 10)                            # glow
        h = 1 if blink else 6
        pygame.draw.ellipse(s, (165, 243, 252), (x - 5, y - h, 10, h * 2))
    pygame.draw.circle(s, (34, 197, 94) if int(t * 2) % 2 else (21, 128, 61), (cx, cy + 17), 4)  # chest light
    pygame.draw.line(s, (100, 116, 139), (cx, cy - 25), (cx, cy - 37), 3)          # antenna
    pygame.draw.circle(s, (248, 113, 113) if int(t * 3) % 2 else (153, 27, 27), (cx, cy - 39), 5)
    if carrying:
        draw_box(s, pygame.Rect(cx - 14, cy + 4, 28, 21))                          # cargo


class Renderer:
    """Draws one game. It keeps track of animation: smooth movement,
    a fading trail, sparkles and screen shake."""

    def __init__(self, env):
        self.env = env
        self.t = 0.0
        self.shake = 0.0
        self.reset_view()

    def reset_view(self):
        """Call when a new game starts."""
        self.px, self.py = self.center(self.env.robot_pos)
        self.facing = (1, 0)
        self.trail, self.particles, self.flash = [], [], 0.0

    def center(self, pos):
        """Pixel centre of a grid square given as (row, col)."""
        return BX + pos[1] * CELL + CELL / 2, BY + pos[0] * CELL + CELL / 2

    def on_step(self, action, info):
        """Called after every move: start effects for what just happened."""
        self.facing = MOVES[int(action)]
        self.trail = (self.trail + [self.env.robot_pos])[-10:]
        ev = info["event"]
        if ev == "pickup":
            self.burst(self.center(self.env.package_pos), [(250, 204, 21), (253, 224, 71)], 25)
        elif ev == "deliver":
            self.burst(self.center(self.env.goal_pos),
                       [(74, 222, 128), (250, 204, 21), (56, 189, 248), (244, 114, 182)], 90)
        elif ev == "bump":
            self.shake, self.flash = 0.25, 0.3

    def burst(self, pos, colors, n):
        for _ in range(n):
            a, v = random.uniform(0, 6.28), random.uniform(80, 320)
            self.particles.append([pos[0], pos[1], math.cos(a) * v, math.sin(a) * v - 120,
                                   random.uniform(0.5, 1.1), random.choice(colors)])

    def update(self, dt):
        """Move animation forward by dt seconds."""
        self.t += dt
        tx, ty = self.center(self.env.robot_pos)
        k = 1 - math.exp(-18 * dt)                    # smooth glide toward the target square
        self.px += (tx - self.px) * k
        self.py += (ty - self.py) * k
        self.shake = max(0.0, self.shake - dt)
        self.flash = max(0.0, self.flash - dt)
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 500 * dt                          # gravity
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

    def offset(self):
        """Random little screen offset while shaking (after a bump)."""
        a = int(self.shake * 24)
        return (random.randint(-a, a), random.randint(-a, a)) if a else (0, 0)

    # ------------------------------------------------------------ drawing
    def draw(self, s):
        env = self.env
        self._board(s)
        self._goal(s)
        for i, pos in enumerate(self.trail):          # fading footprints
            alpha_circle(s, (34, 211, 238), 25 + i * 9, self.center(pos), 6)
        if not env.carrying:
            self._package(s)
        draw_robot(s, self.px, self.py, self.t, self.facing, env.carrying)
        if self.flash > 0:                            # red flash on a bump
            r = pygame.Rect(0, 0, CELL, CELL)
            r.center = self.center(env.robot_pos)
            panel(s, r, (239, 68, 68), int(self.flash * 400), 10)
        for p in self.particles:
            alpha_circle(s, p[5], int(255 * min(1, p[4])), (p[0], p[1]), 4)

    def _board(self, s):
        n = self.env.size
        size = n * CELL
        frame = pygame.Rect(BX - 14, BY - 14, size + 28, size + 28)
        pygame.draw.rect(s, (6, 10, 20), frame.move(0, 8), border_radius=20)      # shadow
        pygame.draw.rect(s, (51, 65, 85), frame, border_radius=20)                # frame
        pygame.draw.rect(s, (250, 204, 21), frame.inflate(-10, -10), 2, border_radius=16)
        for r in range(n):
            for c in range(n):
                rect = pygame.Rect(BX + c * CELL, BY + r * CELL, CELL, CELL).inflate(-3, -3)
                base = (71, 85, 105) if (r + c) % 2 == 0 else (62, 76, 97)
                pygame.draw.rect(s, base, rect, border_radius=8)
                pygame.draw.line(s, (100, 116, 139), (rect.x + 8, rect.y + 2), (rect.right - 8, rect.y + 2), 2)
                for cx, cy in ((rect.x + 8, rect.y + 8), (rect.right - 8, rect.y + 8),
                               (rect.x + 8, rect.bottom - 8), (rect.right - 8, rect.bottom - 8)):
                    pygame.draw.circle(s, (51, 65, 85), (cx, cy), 2)              # little rivets

    def _goal(self, s):
        cx, cy = self.center(self.env.goal_pos)
        pulse = (math.sin(self.t * 3) + 1) / 2
        for i in range(3):                                                         # pulsing rings
            alpha_circle(s, (74, 222, 128), int(90 - i * 25), (cx, cy), int(34 + i * 10 + pulse * 6), 3)
        pad = pygame.Rect(0, 0, CELL - 22, CELL - 22)
        pad.center = (cx, cy)
        pygame.draw.rect(s, (21, 128, 61), pad, border_radius=12)
        pygame.draw.rect(s, (74, 222, 128), pad, 3, border_radius=12)
        pygame.draw.rect(s, (240, 253, 244), (cx - 15, cy - 6, 30, 22), border_radius=2)    # house
        pygame.draw.polygon(s, (22, 101, 52), [(cx - 20, cy - 5), (cx, cy - 24), (cx + 20, cy - 5)])
        pygame.draw.rect(s, (21, 128, 61), (cx - 4, cy + 2, 8, 14))                          # door
        text(s, "GOAL", (cx, pad.bottom + 1), 13, (187, 247, 208), True)

    def _package(self, s):
        cx, cy = self.center(self.env.package_pos)
        bob = math.sin(self.t * 4) * 3
        pygame.draw.ellipse(s, (10, 15, 28), (cx - 24, cy + 18, 48, 12))
        alpha_circle(s, (250, 204, 21), int(40 + 30 * (math.sin(self.t * 4) + 1) / 2), (cx, cy), 36)  # glow
        box = pygame.Rect(0, 0, 46, 38)
        box.center = (cx, cy + bob)
        draw_box(s, box)
        pygame.draw.rect(s, (255, 255, 255), (box.x + 6, box.y + 7, 14, 10), border_radius=2)   # label
