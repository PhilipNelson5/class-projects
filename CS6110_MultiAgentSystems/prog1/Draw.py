import pygame
import math
from typing import Tuple

import Snek, Laser
import Colors as c

Color = Tuple[int, int, int]
Point = Tuple[int, int]

def snek(snek: Snek, dispSurf: pygame.Surface, size: int) -> None:
    for body in snek.locs:
        x, y = (n * size for n in body)

        outside = pygame.Rect(x, y, size, size)
        pygame.draw.rect(dispSurf, snek.outerColor, outside)

        inside = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        pygame.draw.rect(dispSurf, snek.innerColor, inside)


def apple(apple: Point, dispSurf: pygame.Surface, size: int) -> None:
    radius: float = math.floor(size/2.5)
    x, y = (n * size for n in apple)
    xcenter, ycenter = (n * size + math.floor(size / 2) for n in apple)
    pygame.draw.circle(dispSurf, c.RED, (xcenter, ycenter), radius)

    #appleRect = pygame.Rect(x, y, size, size)
    #pygame.draw.rect(dispSurf, c.RED, appleRect)
    

def laser(laser: Laser, dispSurf: pygame.Surface, size: int) -> None:
    for loc in laser.locs:
        x, y = (n * size for n in loc)

        inside = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        pygame.draw.rect(dispSurf, c.RED, inside)


def stone(stone: Point, dispSurf: pygame.Surface, size: int) -> None:
    x, y = (n * size for n in stone)

    outside = pygame.Rect(x, y, size, size)
    pygame.draw.rect(dispSurf, c.PURPLE, outside)

    inside = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
    pygame.draw.rect(dispSurf, c.BLACK, inside)


def startScreen(dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int, winHeight: int, deg1: int, deg2: int) -> None:
    titleFont = pygame.font.Font('freesansbold.ttf', 65)
    titleSurf1 = titleFont.render('Nope Ropes', True, c.RED, c.YELLOW)
    titleSurf2 = titleFont.render('Danger Noodles', True, c.GREEN)

    rotatedSurf1 = pygame.transform.rotate(titleSurf1, deg1)
    rotatedRect1 = rotatedSurf1.get_rect()
    rotatedRect1.center = (math.floor(winWidth / 2), math.floor(winHeight / 2))
    dispSurf.blit(rotatedSurf1, rotatedRect1)

    rotatedSurf2 = pygame.transform.rotate(titleSurf2, deg2)
    rotatedRect2 = rotatedSurf2.get_rect()
    rotatedRect2.center = (math.floor(winWidth / 2), math.floor(winHeight / 2))
    dispSurf.blit(rotatedSurf2, rotatedRect2)

    pressKeyMsg(dispSurf, font, winWidth, winHeight)


def score(score: int, n: int, dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int) -> None:
    scoreSurf = font.render(f'Player {n}: {score}', True, c.WHITE)
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (winWidth - 120, 20 * n)
    dispSurf.blit(scoreSurf, scoreRect)


def fill(dispSurf: pygame.Surface, color: Color):
    dispSurf.fill(color)


def grid(dispSurf: pygame.Surface, size: int, winWidth: int, winHeight: int) -> None:
    for x in range(0, winWidth, size): # draw vertical lines
        pygame.draw.line(dispSurf, c.DARKGRAY, (x, 0), (x, winHeight))
    for y in range(0, winHeight, size): # draw horizontal lines
        pygame.draw.line(dispSurf, c.DARKGRAY, (0, y), (winWidth, y))


def pressKeyMsg(dispSurf: pygame.Surface, font: pygame.font.Font, winWidth: int, winHeight: int) -> None:
    pressKeySurf = font.render('Press a key to play.', True, c.YELLOW)
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
