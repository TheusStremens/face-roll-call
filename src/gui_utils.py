import cv2


def get_text_position(text, rect):
    w, h = cv2.getTextSize(text, 0, fontScale=1.0, thickness=2)[0]
    x = (rect[2]/2) - (w/2)
    y = (rect[3]/2) - (h/2) + h
    return [rect[0] + int(x), rect[1] + int(y)]


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    default_args = {"color": color, "thickness": thickness, "lineType": cv2.LINE_AA}
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), **default_args)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), **default_args)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, **default_args)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), **default_args)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), **default_args)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, **default_args)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), **default_args)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), **default_args)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, **default_args)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), **default_args)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), **default_args)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, **default_args)


def rect_contains(rect, pt):
    check = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return check
