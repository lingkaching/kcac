import gymGame

class Collider2D(gymGame.GameComponent):
    def __init__(self):
        super().__init__()

    def isTouching(self, other):
        raise NotImplementedError()

class BoxCollider2D(Collider2D):
    def __init__(self, w=1, h=1):
        super().__init__()
        self.width = w
        self.height = h

    def isTouching(self, other: Collider2D):
        if isinstance(other, BoxCollider2D):
            xCollision = abs(self.gameObject.position[0] - other.gameObject.position[0]) <= (self.width + other.width) / 2
            yCollision = abs(self.gameObject.position[1] - other.gameObject.position[1]) <= (self.width + other.width) / 2
            return xCollision and yCollision
        else:
            raise NotImplementedError()

    