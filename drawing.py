import pygame
import numpy as np
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, DRAWING_COLOR, FONT_SIZE, FONT, FONT_COLOR

FILLED = 1
EMPTY = 0


class DrawingGrid:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.square_width = SCREEN_WIDTH / width
        self.square_height = SCREEN_HEIGHT / height
        self.grid = np.zeros(shape=(width, height))

    def handle_event(self, event):

        if event.type == pygame.MOUSEMOTION:
            # check for buttons currently being pressed
            buttons = event.buttons
            if buttons[0]:  # left mouse button
                self.draw(event.pos)
            elif buttons[2]:  # right mouse button
                self.erase(event.pos)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                self.grid.fill(EMPTY)

    def draw(self, mouse_pos):
        x, y = self.get_grid_coords(*mouse_pos)
        # set all neighbor squares as well
        for i in range(-1, 1):
            for j in range(-1, 1):
                self.set_square(x + i, y + j, FILLED)

    def erase(self, mouse_pos):
        x, y = self.get_grid_coords(*mouse_pos)
        self.set_square(x, y, EMPTY)

    def set_square(self, x, y, state):
        x, y = int(x), int(y)
        self.grid[x][y] = state

    def get_grid_coords(self, x, y):
        return x // self.square_width, y // self.square_height

    def display(self, screen):
        for y, row in enumerate(self.grid, 1):
            for x, square in enumerate(row):
                if square != EMPTY:
                    rect = pygame.Rect(y * self.square_height, x * self.square_width,
                                       self.square_height, self.square_width)
                    pygame.draw.rect(screen, DRAWING_COLOR, rect)

    def get_grid_for_predicting(self):
        return np.flip(np.rot90(self.grid, 1), axis=0)


def draw_texts_top_right(screen, *texts):
    font = pygame.font.Font(FONT, FONT_SIZE)
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, FONT_COLOR)
        text_position = SCREEN_WIDTH - 20 - text_surface.get_width(), \
                        50 + (FONT_SIZE + 10) * i
        screen.blit(text_surface, text_position)
