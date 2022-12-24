import threading


def thread(set_deamon = False):
    """
    多线程装饰器
    :param set_deamon:
    :return:
    """
    def thread_decorator(func):
        def wrapper(*args, **kwargs):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.setDaemon(set_deamon)
            thread.start()
            return thread
        return wrapper
    return thread_decorator