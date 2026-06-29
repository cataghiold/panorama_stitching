import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setari globale pentru detectarea caracteristicilor si potrivirea lor
MAX_FEATURES = 500  # Numarul maxim de caracteristici ce vor fi detectate
GOOD_MATCH_PERCENT = 0.15  # Procentul de potriviri bune care vor fi pastrate

def load_images():
    #Incarca imaginile din caile specificate
    paths = [
        r'C:\Users\Cata\Desktop\Imagini\1.png',
        r'C:\Users\Cata\Desktop\Imagini\2.png',
        r'C:\Users\Cata\Desktop\Imagini\3.png',
    ]
    images = []
    for path in paths:
        # Citeste fiecare imagine, o adauga intr-o lista si le returneaza
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {path}")
        images.append(img)
    return images

def detect_and_match_keypoints(img1, img2):
    # Detecteaza puncte cheie si le potriveste intre 2 imagini
    # Creeaza un detector ORB (Oriented FAST and Rotated BRIEF) pentru a extrage keypoints,
    orb = cv2.ORB_create(MAX_FEATURES)

    # Detecteaza keypoints si calculeaza descriptorii pentru fiecare imagine
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Verifica daca descriptorii au fost calculati corect
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors could not be computed for one or both images.")

    # Creeaza un potrivitor de descriptori (Brute Force Hamming pentru descriptorii ORB)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sorteaza potrivirile in functie de distanta (cele mai bune primele)
    matches = sorted(matches, key=lambda x: x.distance)

    # Pastreaza doar un procentaj dintre cele mai bune potriviri
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    return keypoints1, keypoints2, matches

def stitch_images(img1, img2):
    #Lipeste doua imagini folosind homografie si transformare de perspectiva.
    # Detecteaza keypoints si potrivirile dintre imagini
    keypoints1, keypoints2, matches = detect_and_match_keypoints(img1, img2)

    # Extrage coordonatele punctelor potrivite intre cele doua imagini
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Calculeaza matricea de homografie folosind metoda RANSAC
    h, status = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Verifica daca homografia este valida
    if h is None:
        raise ValueError("Homography could not be computed.")
    if np.linalg.matrix_rank(h) < 3:
        raise ValueError("Homography matrix is invalid.")

    # Obtine dimensiunile imaginilor pentru a calcula dimensiunea finala a canvas-ului
    height, width = img1.shape[:2]
    img2_height, img2_width = img2.shape[:2]

    # Calculeaza colturile imaginii a doua dupa aplicarea homografiei
    corners_img2 = np.array([[0, 0], [0, img2_height], [img2_width, 0], [img2_width, img2_height]], dtype=np.float32)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2[None, :, :], h)[0]

    # Combina toate colturile pentru a determina dimensiunile imaginii finale
    all_corners = np.vstack((np.array([[0, 0], [0, height], [width, 0], [width, height]]), warped_corners_img2))

    # Calculeaza limitele imaginii finale
    [x_min, y_min] = np.int32(all_corners.min(axis=0))
    [x_max, y_max] = np.int32(all_corners.max(axis=0))

    # Creeaza un canvas suficient de mare pentru a include ambele imagini
    translation_dist = [-x_min, -y_min]
    result_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

    # Translateaza imaginea 2 pentru a o aduce in coordonatele corecte
    h_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]], dtype=np.float32)
    result_img = cv2.warpPerspective(img2, h @ h_translation, (result_img.shape[1], result_img.shape[0]))

    # Adauga imaginea 1 pe canvas
    result_img[translation_dist[1]:translation_dist[1] + height, translation_dist[0]:translation_dist[0] + width] = img1

    return result_img

def crop_to_rectangle(panorama):
    #Decupeaza panorama pentru a elimina marginile negre.
    # Coordonatele predefinite pentru decupare
    x_min, y_min = 100, 120
    x_max, y_max = 2700, 1200

    # Verifica limitele dimensiunii imaginii pentru decupare
    height, width, _ = panorama.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # Decupeaza imaginea
    cropped_panorama = panorama[y_min:y_max, x_min:x_max]

    return cropped_panorama

def main():
    # Punctul de intrare principal al programului.
    # Incarca imaginile din fisiere
    try:
        images = load_images()
    except FileNotFoundError as e:
        print(e)
        return

    # Lipeste imaginile una cate una
    try:
        stitched_image = images[0]
        for i in range(1, len(images)):
            stitched_image = stitch_images(stitched_image, images[i])
    except ValueError as e:
        print(e)
        return

    # Decupeaza panorama finala pentru a elimina marginile negre
    final_panorama = crop_to_rectangle(stitched_image)

    # Afiseaza imaginile originale, imaginea lipita si panorama finala
    plt.figure(figsize=(20, 10))

    for i, img in enumerate(images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {i+1}")
        plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Image")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
    plt.title("Final Panorama (Cropped)")
    plt.axis("off")
    plt.show()

main()
