import pygame
import math
import rendering as r
from typing import Tuple

import Colors as c

Color = Tuple[int, int, int]
Point = Tuple[int, int]


def startScreen(dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int, winHeight: int, deg1: int, deg2: int) -> None:
    titleFont = pygame.font.Font('freesansbold.ttf', 65)
    titleSurf1 = titleFont.render('Roomba', True, c.FIREBRICK, c.KHAKI)
    titleSurf2 = titleFont.render('Roomba', True, c.DARK_SLATE_GRAY)

    rotatedSurf1 = pygame.transform.rotate(titleSurf1, deg1)
    rotatedRect1 = rotatedSurf1.get_rect()
    rotatedRect1.center = (math.floor(winWidth / 2), math.floor(winHeight / 2))
    dispSurf.blit(rotatedSurf1, rotatedRect1)

    rotatedSurf2 = pygame.transform.rotate(titleSurf2, deg2)
    rotatedRect2 = rotatedSurf2.get_rect()
    rotatedRect2.center = (math.floor(winWidth / 2), math.floor(winHeight / 2))
    dispSurf.blit(rotatedSurf2, rotatedRect2)

    pressKeyMsg(dispSurf, font, winWidth, winHeight)


def score(score: int, n: int, font: pygame.font.Font, winWidth: int) -> None:
    scoreSurf = font.render(f'Roomba {n}: {score}', True, c.BLACK)
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (winWidth - 160, 20 * n)
    r.screen.blit(scoreSurf, scoreRect)


def fill(color: Color):
    r.screen.fill(color)


def grid(dispSurf: pygame.Surface, size: int, winWidth: int, winHeight: int) -> None:
    for x in range(0, winWidth, size): # draw vertical lines
        pygame.draw.line(dispSurf, c.DIM_GRAY, (x, 0), (x, winHeight))
    for y in range(0, winHeight, size): # draw horizontal lines
        pygame.draw.line(dispSurf, c.DIM_GRAY, (0, y), (winWidth, y))


def pressKeyMsg(dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int, winHeight: int) -> None:
    pressKeySurf = font.render('Press a key to play.', True, c.STEEL_BLUE)
    pressKeyRect = pressKeySurf.get_rect()
    pressKeyRect.topleft = (winWidth - 200, winHeight - 30)
    dispSurf.blit(pressKeySurf, pressKeyRect)


def gameOver(dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int, winHeight: int) -> None:
    gameSurf = font.render('Game', True, c.WHITE)
    overSurf = font.render('Over', True, c.WHITE)
    gameRect = gameSurf.get_rect()
    overRect = overSurf.get_rect()
    gameRect.midtop = (math.floor(winWidth / 2), 10)
    overRect.midtop = (math.floor(winWidth / 2), gameRect.height + 10 + 25)

    dispSurf.blit(gameSurf, gameRect)
    dispSurf.blit(overSurf, overRect)
