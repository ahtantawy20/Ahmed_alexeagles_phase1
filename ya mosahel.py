import cv2
import numpy as np

# ------- binarize function -------
def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

# ------- count gear teeth -------
def count_teeth(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 2000:  # threshold distance
            count += 1
    return count

# ------- radius profile -------
def get_radius_profile(contour, center):
    radii = []
    for point in contour:
        x, y = point[0]
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        radii.append(r)
    return np.array(radii)

# ------- analyze gear -------
def analyze_gear(img, name="sample", ideal_teeth=None, ideal_radius=None, ideal_inner_area=None):
    bin_img = binarize(img)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, {"status": "no gear detected"}

    # sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt_outer = contours[0]

    # find center
    M = cv2.moments(cnt_outer)
    if M["m00"] == 0:
        cx, cy = 0, 0
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    # count teeth
    num_teeth = count_teeth(cnt_outer)

    # inner hole check
    inner_hole = None
    inner_status = "no inner hole"
    inner_area = 0
    if len(contours) > 1:
        for c in contours[1:]:
            area = cv2.contourArea(c)
            if area > 200:  # ignore noise
                inner_hole = c
                inner_area = area
                inner_status = "inner hole detected"
                break

    # radius profile
    sample_radius = get_radius_profile(cnt_outer, center)
    avg_radius = np.mean(sample_radius)

    # compare with ideal
    hole_comparison = "N/A"
    if ideal_inner_area is not None and inner_area > 0:
        # Calculate the ratio of areas
        area_diff = abs(inner_area - ideal_inner_area) / ideal_inner_area
        if area_diff < 0.1:
            hole_comparison = "inner hole same size"
        elif inner_area > ideal_inner_area:
            hole_comparison = "inner hole bigger"
        else:
            hole_comparison = "inner hole smaller"

    # draw results
    debug_img = img.copy()
    cv2.drawContours(debug_img, [cnt_outer], -1, (0, 255, 0), 2)
    if inner_hole is not None:
        cv2.drawContours(debug_img, [inner_hole], -1, (0, 255, 0), 2)

    # put text results
    text_y = 30
    cv2.putText(debug_img, f"{name}: {num_teeth} teeth", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    text_y += 25
    cv2.putText(debug_img, f"Hole: {inner_status}", (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if hole_comparison != "N/A":
        text_y += 25
        cv2.putText(debug_img, f"Compare: {hole_comparison}", (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    results = {
        "teeth": num_teeth,
        "inner_status": inner_status,
        "hole_comparison": hole_comparison
    }
    return debug_img, results


# ------- main -------
ideal = cv2.imread("ideal.jpg")
ideal_debug, ideal_results = analyze_gear(ideal, "ideal")
ideal_teeth = ideal_results["teeth"]
ideal_inner_area = cv2.contourArea(cv2.findContours(binarize(ideal), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][1])  # Ideal inner hole area

print(f"Ideal gear -> Teeth={ideal_teeth}, Inner hole area={ideal_inner_area}")

samples = ["sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg", "sample6.jpg"]

all_results = [ideal_debug]

for fname in samples:
    img = cv2.imread(fname)
    debug_img, results = analyze_gear(img, fname, ideal_teeth, ideal_radius=None, ideal_inner_area=ideal_inner_area)
    print(f"{fname} -> Teeth={results['teeth']}, Hole: {results['hole_comparison']}")
    all_results.append(debug_img)

# ---- Show all in grid ----
cols = 3  # عدد الأعمدة
rows = (len(all_results) + cols - 1) // cols

h, w, _ = all_results[0].shape
while len(all_results) < rows * cols:
    all_results.append(np.zeros((h, w, 3), dtype=np.uint8))

grid_rows = []
for r in range(rows):
    row_imgs = all_results[r*cols:(r+1)*cols]
    grid_rows.append(np.hstack(row_imgs))

combined = np.vstack(grid_rows)
combined = cv2.resize(combined, (0, 0), fx=0.6, fy=0.6)

cv2.imshow("Gear Analysis", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
