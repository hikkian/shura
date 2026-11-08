FLAGS = {"loyalty_points": True, "reviews": True, "gift_wrap": False}


def is_enabled(name):
    return FLAGS.get(name, False)
