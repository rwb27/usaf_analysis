
from collections import namedtuple

class PiResolution(namedtuple('PiResolution', ('width', 'height'))):
    """
    A :func:`~collections.namedtuple` derivative which represents a resolution
    with a :attr:`width` and :attr:`height`.

    .. attribute:: width

        The width of the resolution in pixels

    .. attribute:: height

        The height of the resolution in pixels

    .. versionadded:: 1.11
    """

    __slots__ = () # workaround python issue #24931

    def pad(self, width=32, height=16):
        """
        Returns the resolution padded up to the nearest multiple of *width*
        and *height* which default to 32 and 16 respectively (the camera's
        native block size for most operations). For example:

        .. code-block:: pycon

            >>> PiResolution(1920, 1080).pad()
            PiResolution(width=1920, height=1088)
            >>> PiResolution(100, 100).pad(16, 16)
            PiResolution(width=128, height=112)
            >>> PiResolution(100, 100).pad(16, 16)
            PiResolution(width=112, height=112)
        """
        return PiResolution(
            width=((self.width + (width - 1)) // width) * width,
            height=((self.height + (height - 1)) // height) * height,
            )

    def transpose(self):
        """
        Returns the resolution with the width and height transposed. For
        example:

        .. code-block:: pycon

            >>> PiResolution(1920, 1080).transpose()
            PiResolution(width=1080, height=1920)
        """
        return PiResolution(self.height, self.width)

    def __str__(self):
        return '%dx%d' % (self.width, self.height)
