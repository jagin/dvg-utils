class Observable:
    """ Observable pattern"""
    def __init__(self):
        self._events = {}

    def register(self, event, observer, callback):
        if event not in self._events:
            self._events[event] = {}
        self._events[event][observer] = callback

        return self

    def unregister(self, event, observer):
        del self._events[event][observer]

    def notify(self, event, *args, **kwargs):
        if event in self._events:
            for observer, callback in self._events[event].items():
                callback(*args, **kwargs)


