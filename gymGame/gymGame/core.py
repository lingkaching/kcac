import gymGame
import numpy as np
import gym
import random
from typing import List, Set, TypeVar  # noqa: F401


class GameObject:
    def __init__(self):
        self.name = "GameObject"
        self.isActive = True
        self.layer = 'general'
        self._components = []  # type: List[GameComponent]
        self._children = set()  # type: Set[GameObject]
        self._parent = None  # type: GameObject
        self.position = np.array([0.0, 0.0, 0.0])
        self.collider2D = None  # type: gymGame.Collider2D
        self.scene = None  # type: Scene

    def activate(self):
        self.isActive = True
        for c in self._components:
            c.onEnable()
        for child in self._children:
            child.activate()

    def deactivate(self):
        self.isActive = False
        for c in self._components:
            c.onDisable()
        for child in self._children:
            child.deactivate()

    def addComponent(self, c):
        self._components.append(c)
        c.gameObject = self
        if isinstance(c, gymGame.Collider2D):
            self.collider2D = c

    def getComponent(self, typ: type, tag=None):
        # TODO: make this more efficient
        return next(filter(lambda c: isinstance(c, typ) and (tag is None or c.tag == tag), self._components), None)

    def setPosition(self, newPosition):
        change = newPosition - self.position
        self.move(change)

    def move(self, change):
        self.position += change
        for child in self._children:
            child.move(change)

    def setParent(self, parentGameObject):
        self.removeParent()
        self._parent = parentGameObject
        parentGameObject._children.add(self)

    def removeParent(self):
        if self._parent is not None:
            self._parent._children.remove(self)
            self._parent = None


class GameComponent:
    count = 0

    def __init__(self):
        self._isEnabled = True
        self.gameObject = None  # type: GameObject
        self._startCalled = False
        self._awakeCalled = False
        self._enableCalled = False
        self.tag = None
        self._init_order = GameComponent.count
        GameComponent.count += 1

    def _awake(self):
        if not self._awakeCalled:
            self.awake()
            self._awakeCalled = True

    def awake(self):
        # called when object is initialized irrespective of whether it is enabled or not
        # in beginning of scene, it is called after initializing all objects only
        pass

    def _start(self):
        if not self._startCalled:
            self.start()
            self._startCalled = True

    def start(self):
        # called once after component is enabled and active
        pass

    def update(self):
        # called every frame if component is enabled and active
        pass

    def _onEnable(self):
        if not self._enableCalled:
            self.onEnable()
            self._enableCalled = True

    def onEnable(self):
        # called when object is enabled
        pass

    def onDisable(self):
        # called when object is disabled or destroyed
        pass

    def enable(self):
        if not self._isEnabled:
            self._isEnabled = True
            self.onEnable()
            self._enableCalled = True
            self._start()

    def disable(self):
        if self._isEnabled:
            self._isEnabled = False
            self.onDisable()


class Scene(gym.Env):

    def __init__(self):
        super().__init__()
        self._gameObjects = set()  # type: Set[GameObject]
        self._dontDestroyOnLoadObjects = set()  # type: Set[GameObject]
        self._isRunning = False
        self.random = random.Random()
        self.nprandom = np.random.RandomState()

    def instantiate(self, cls, position=None) -> GameObject:
        if issubclass(cls, GameObject):
            obj = cls()  # type: GameObject
            # print('Created GameObject ' + obj.name)
            if position is not None:
                obj.setPosition(position)
            self._gameObjects.add(obj)
            obj.scene = self
            if self._isRunning:
                # run awake for all components
                for c in obj._components:
                    c._awake()
                    if obj.isActive and c._isEnabled:
                        c.onEnable()
            return obj
        else:
            raise TypeError()

    def findObjectByName(self, name):
        return next(filter(lambda o: o.name == name, self._gameObjects), None)

    def dontDestroyOnLoad(self, gameObj: GameObject):
        print('GameObject {0} marked as dontDestroyOnLoad'.format(
            gameObj.name))
        self._dontDestroyOnLoadObjects.add(gameObj)

    def destroy(self, gameObj: GameObject):
        # print('Destroying GameObject ' + gameObj.name)
        if gameObj.isActive:
            for c in gameObj._components:
                c.onDisable()
        self._gameObjects.remove(gameObj)
        self._dontDestroyOnLoadObjects.discard(gameObj)
        for c in gameObj._components:
            if hasattr(type(c), 'instance'):
                # print('Static reference found')
                if getattr(type(c), 'instance') == c:
                    # print('The reference set to null on destroying the object')
                    setattr(type(c), 'instance', None)

    def _destroyObjects(self, destroyAll=False):
        if destroyAll:
            toDestroy = self._gameObjects.copy()
        else:
            toDestroy = self._gameObjects - self._dontDestroyOnLoadObjects
        for obj in toDestroy:
            self.destroy(obj)
        self._isRunning = False
        GameComponent.count = 0

    def reset(self):
        self._isRunning = True
        # run all awakes and enabled
        self._executeInOrder(self._gameObjects, lambda c: c._awake())
        self._executeInOrder(self._gameObjects, lambda c: c._onEnable(),
                             objFilter=lambda obj: obj.isActive, compFilter=lambda c: c._isEnabled)

        # call starts
        self._executeInOrder(self._gameObjects, lambda c: c._start(),
                             objFilter=lambda obj: obj.isActive, compFilter=lambda c: c._isEnabled)

        return None

    def _executeInOrder(self, objects, fn, objFilter=lambda obj: True, compFilter=lambda c: True):
        for c in self.getComponentsInExecutionOrderFromObjects(objects, objFilter, compFilter):
            fn(c)

    def getComponentsInExecutionOrderFromObjects(self, objects, objFilter=lambda obj: True, compFilter=lambda c: True):
        components = []
        for obj in filter(objFilter, objects):
            components.extend(filter(compFilter, obj._components))
        return self._inExecutionOrder(components)

    def getAllComponentsInExecutionOrder(self, objFilter=lambda obj: True, compFilter=lambda c: True):
        return self.getComponentsInExecutionOrderFromObjects(self._gameObjects, objFilter, compFilter)

    def _inExecutionOrder(self, components):
        for c in components:
            if not hasattr(type(c), 'executionOrder'):
                type(c).executionOrder = 0
        return sorted(components, key=lambda c: (type(c).executionOrder, c._init_order))

    def step(self, action):
        self._executeInOrder(self._gameObjects, lambda c: c.update(),
                             objFilter=lambda obj: obj.isActive, compFilter=lambda c: c._isEnabled)
        return None, 0, False, {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    def seed(self, seed=None):
        self.random.seed(seed)
        self.nprandom.seed(seed)

    def close(self):
        self._destroyObjects(destroyAll=True)
        self._isRunning = False


def set_execution_order(component_classes: List[GameComponent]):
    order = 0
    for c in component_classes:
        c.executionOrder = order
        order += 100
