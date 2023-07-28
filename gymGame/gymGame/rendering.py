import gymGame
import pygame
import numpy as np
from typing import List, Dict, Set


class SimpleSprite(gymGame.GameComponent):
    def __init__(self, sprite, w=1, h=1, static=False):
        super().__init__()
        self.setSize([w, h])
        self.setRotation(0)
        self.sprite = sprite  # type: pygame.Surface
        self.static = static

    def awake(self):
        self.camera = self.gameObject.scene.findObjectByName(
            'Main Camera').getComponent(gymGame.Camera)  # type: gymGame.Camera

    def setSize(self, size):
        self.size = size
        self.transform_changed = True

    def setRotation(self, radians):
        self.rotation_degrees = radians * 180 / np.pi
        self.transform_changed = True

    def load(filename):
        return pygame.image.load(filename)  # .convert_alpha()

    def onEnable(self):
        if self.camera._isEnabled:
            self.camera.spritesBatch.add(self)

    def update(self):
        if self.camera._isEnabled:
            self.camera.spritesBatch.add(self)


class Camera(gymGame.GameComponent):
    def __init__(self, renderingSurface, fov, backgroundColor=(0, 0, 0)):
        super().__init__()
        self.spritesBatch = set()  # type: Set[SimpleSprite]
        self._dirtyRects = []
        self.setFov(fov)
        self.setRenderingSurface(renderingSurface)
        self.backgroundColor = backgroundColor
        self.staticBackground = None
        self._transformedSprites = {}  # type: Dict[SimpleSprite, pygame.Surface]
        self.latestFrame = np.zeros([210, 150, 3], dtype=np.uint8)
        self.auto_render = False

    def createRenderingSurface(resolution):
        return pygame.Surface(resolution)  # .convert_alpha()

    def setRenderingSurface(self, surface):
        self.surface = surface
        self.resolution = (surface.get_width(), surface.get_height())

    def setFov(self, fov):
        self.fov = fov
        self._map_bounds = [[-fov[0] / 2, -fov[1] / 2], [fov[0] / 2, fov[1] / 2]]

    def _getCoordinatesOnSurface(self, position):
        x = (position[0] - self._map_bounds[0][0]) * \
            self.resolution[0] // (self._map_bounds[1][0] - self._map_bounds[0][0])
        y = (position[1] - self._map_bounds[0][1]) * \
            self.resolution[1] // (self._map_bounds[1][1] - self._map_bounds[0][1])
        return [x, self.resolution[1] - y]

    def _getSizeOnSurface(self, size):
        return (int((size[0] * self.resolution[0]) / self.fov[0]), int((size[1] * self.resolution[1]) / self.fov[1]))

    def _getTransformedSprite(self, ss: SimpleSprite):
        if ss in self._transformedSprites and not ss.transform_changed:
            return self._transformedSprites[ss]
        else:
            size = self._getSizeOnSurface(ss.size)
            transformed = pygame.transform.smoothscale(ss.sprite, size)
            transformed = pygame.transform.rotate(transformed, ss.rotation_degrees)
            self._transformedSprites[ss] = transformed
            ss.transform_changed = False
            return transformed

    def update(self):
        pass

    def _drawSpriteComponent(self, sc):
        upperLeft = (sc.gameObject.position[0] - sc.size[0] / 2,
                     sc.gameObject.position[1] + sc.size[1] / 2)
        dest = self._getCoordinatesOnSurface(upperLeft)
        transformedSprite = self._getTransformedSprite(sc)
        rect = (dest, (transformedSprite.get_width(), transformedSprite.get_height()))
        self.surface.blit(transformedSprite, dest)
        return rect

    def _clear(self):
        if self.staticBackground is None:
            self.surface.fill(self.backgroundColor)
            staticSprites = filter(
                lambda sc: sc.static and sc._isEnabled and sc.gameObject.isActive, self.spritesBatch)
            for sc in staticSprites:
                self._drawSpriteComponent(sc)
            self.staticBackground = self.surface.copy()
        else:
            for r in self._dirtyRects:
                # self.surface.fill(self.backgroundColor, r)
                self.surface.blit(self.staticBackground, r, r)
        self._dirtyRects.clear()

    def render(self):
        self._clear()
        for sc in filter(lambda s: not s.static and s._isEnabled and s.gameObject.isActive, self.spritesBatch):
            rect = self._drawSpriteComponent(sc)
            self._dirtyRects.append(rect)
        self.spritesBatch.clear()

    def getLatestFrame(self):
        self.render()
        # if self.latestFrame is not None:
        self.latestFrame = pygame.surfarray.pixels3d(self.surface).copy()
        # self.latestFrame = np.zeros([210,150,3], dtype=np.uint8)
        # pygame.pixelcopy.surface_to_array(self.latestFrame, self.surface)
        self.latestFrame = np.swapaxes(self.latestFrame, 0, 1)
        return self.latestFrame
