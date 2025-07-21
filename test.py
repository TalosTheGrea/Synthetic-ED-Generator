import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import os
import math
import glob

# Output configuration
OUTPUT_DIR = "generated_eds2"
NUM_IMAGES = 250 # Number of images to generate
TEST_SPLIT_RATIO = 0  # 20% of images will be used for testing
USE_YOLO_FORMAT = False  # Whether to save labels in YOLO format
YOLO_CLASS_ID = 0  # Class ID for text in YOLO format
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create directory structure
TRAIN_IMAGES_DIR = os.path.join(OUTPUT_DIR, "training_images")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_DIR, "training_labels_gt")
TEST_IMAGES_DIR = os.path.join(OUTPUT_DIR, "test_images")
TEST_LABELS_DIR = os.path.join(OUTPUT_DIR, "test_labels_gt")

# Create all directories
for directory in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Canvas configuration
CANVAS_WIDTH = 640  # Increased from 800 for higher DPI
CANVAS_HEIGHT = 640 # Increased from 800 for higher DPI
QUALITY = 80  # Quality setting for output images

# Text configuration
MIN_TEXT_LENGTH = 4    # Minimum length of text
MAX_TEXT_LENGTH = 6    # Maximum length of text
NUMBER_RATIO = 0.5     # Ratio of numbers in the text (e.g., 0.6 means 60% numbers, 40% letters)
FONT_SIZE = 17        # Base font size
FONT_DIR = "fonts"     # Directory containing fonts
VERTICAL_TEXT_SPACING = 1.5  # Spacing between vertical text and rectangle
HORIZONTAL_TEXT_SPACING = 1.5  # Spacing between horizontal text and bottom line
TEXT_OVERLAP_TOLERANCE = 0.01  # Maximum allowed overlap between texts (5%)
VERTICAL_TEXT_BOTTOM_LEFT = True  # Place vertical text at bottom-left instead of center-left
HORIZONTAL_TEXT_BOTTOM_LEFT = True  # Place horizontal text at bottom-left instead of bottom-center

# Multi-row text configuration
MIN_TEXT_ROWS = 1      # Minimum number of rows for horizontal text
MAX_TEXT_ROWS = 4      # Maximum number of rows for horizontal text
TEXT_ROW_SPACING = 2   # Vertical spacing between text rows (in pixels)

# Text splitting configuration
TEXT_SPLIT_PROBABILITY = 0.3  # Probability that a text row will be split into 2 segments
TEXT_SEGMENT_SPACING = 8      # Horizontal spacing between split text segments (in pixels)

# Symbol configuration
USE_SPECIAL_SYMBOLS = False  # Whether to use special symbols in text
SPECIAL_SYMBOL_PROBABILITY = 1  # Probability of adding a special symbol to text (30%)

# Special characters for tolerance box left word
special_char_segu = [
    '\u25B1',  # White Square with Rounded Corners
    '\u25CB',  # White Circle
    '\u232D',  # Cylinder
    '\u2312',  # Arc
    '\u2313',  # Arc with Bullet
    '\u27C2',  # Perpendicular
    '\u2220',  # Angle
    '\u2316',  # Position
    '\u25CE',  # Bullseye
    '\u232F',  # Cylinder with Horizontal Line
    '\u2197',  # North East Arrow
    '\u2330'   # Cylinder with Vertical Line
]

# Additional symbols for text
ADDITIONAL_SYMBOLS = [
    '(', ')',   # Parentheses
    '+',        # Plus
    '−',        # Minus (en dash)
    '±',        # Plus-Minus
    ':',        # Colon
    '/',        # Forward Slash
    '°',        # Degree
    '"',        # Double Quote
    '∅',        # Diameter
    '.'         # Period
]

# Strikethrough configuration
USE_TEXT_STRIKETHROUGH = True  # Whether to use text strikethrough
STRIKETHROUGH_PROBABILITY = 0.3  # Probability of text being struck through
STRIKETHROUGH_DIAGONAL_PROBABILITY = 0.33  # Probability of diagonal strikethrough when strikethrough is applied
STRIKETHROUGH_VERTICAL_PROBABILITY = 0.33  # Probability of vertical strikethrough when strikethrough is applied
STRIKETHROUGH_WIDTH = 1  # Width of strikethrough line
STRIKETHROUGH_COLOR = 'black'  # Color of strikethrough line
STRIKETHROUGH_WORD_BASED = True  # Whether to apply strikethrough per word instead of whole text

# Tolerance box configuration
USE_TOLERANCE_BOX = True  # Whether to use tolerance boxes
TOLERANCE_BOX_MIN_WIDTH = 70  # Minimum rectangle width to show tolerance box
TOLERANCE_BOX_TEXT_PADDING = 1  # Padding inside each tolerance box
TOLERANCE_BOX_MIN_SPACING = 10  # Minimum vertical space between tolerance box and main rectangle
TOLERANCE_BOX_MAX_SPACING = 50  # Maximum vertical space between tolerance box and main rectangle
TOLERANCE_BOX_LINE_WIDTH = 1  # Line width for tolerance box
TOLERANCE_BOX_FONT_SIZE = 20  # Font size for tolerance box text

# Bounding box configuration
BOUNDING_BOX_PADDING = 1.5  # Padding around text for bounding box
SAVE_BOUNDING_BOXES = True  # Whether to save bounding box coordinates
VISUALIZE_BOUNDING_BOXES = False  # Whether to create visualization of bounding boxes
BOUNDING_BOX_COLOR = 'red'  # Color for visualizing bounding boxes
BOUNDING_BOX_LINE_WIDTH = 2  # Line width for visualizing bounding boxes
DISABLE_LEFT_TOLERANCE_BOX_BOUNDING = True  # Whether to disable bounding box for left box in tolerance box

os.makedirs(FONT_DIR, exist_ok=True)

# Get all .ttf fonts from the fonts directory
FONTS = [os.path.basename(f) for f in glob.glob(os.path.join(FONT_DIR, "*.ttf"))]
if not FONTS:
    print("Warning: No .ttf fonts found in the fonts directory. Using default font.")
    FONTS = ["default"]

# Structure configuration
USE_MULTIPLE_STRUCTURES = False  # Can be True (always multiple), False (always single), or "RANDOM" (randomly choose)
# NUM_STRUCTURES will be randomly chosen between 1 and 3 when USE_MULTIPLE_STRUCTURES is True

# Rectangle configuration
MIN_RECTANGLES = 2  # Minimum number of rectangles to generate
MAX_RECTANGLES = 4  # Maximum number of rectangles to generate
STRUCTURE_LENGTH_RATIO = 0.9  # Line will be 90% of the canvas width
MIN_HEIGHT_RATIO = 0.2  # Minimum rectangle height as ratio of canvas height
MAX_HEIGHT_RATIO = 0.7  # Maximum rectangle height as ratio of canvas height
MIN_RECTANGLE_WIDTH = 40  # Minimum width for each rectangle

# Line width configuration
HORIZONTAL_LINE_WIDTH = 1  # Width of horizontal lines
VERTICAL_LINE_WIDTH = 2    # Width of vertical lines

# Dotted line configuration
USE_DOTTED_LINES = True    # Whether to use dotted lines
DOTTED_LINE_PROBABILITY = 0.4  # Probability of a line being dotted
DOTTED_LINE_SPACING = 4    # Space between dots in dotted lines
DOTTED_LINE_LENGTH = 4     # Length of each dot in dotted lines

# Arrow configuration
ARROW_SIZE_RATIO = 0.01  # Arrow size as ratio of structure width
LINE_GAP = 2      # Gap between horizontal and vertical lines

# Mosaic configuration
USE_MOSAIC = True         # Whether to use mosaic pattern
MOSAIC_PROBABILITY = 0.4  # Probability of a rectangle being filled with mosaic
MOSAIC_LINE_SPACING = 8   # Spacing between diagonal lines
MOSAIC_LINE_WIDTH = 1     # Width of diagonal lines
MOSAIC_COLOR = '#555555'     # Color of mosaic lines
MOSAIC_GAP = 2           # Gap between mosaic and horizontal lines

# Diagonal line configuration
USE_DIAGONAL = True      # Whether to use diagonal lines
DIAGONAL_PROBABILITY = 0.3  # Probability of a rectangle having a diagonal line
DIAGONAL_LINE_WIDTH = 1    # Width of diagonal line
DIAGONAL_COLOR = 'black'   # Color of diagonal line
DIAGONAL_TEXT_SPACING = 1   # Base spacing between diagonal text and diagonal line

# Edge diagonal pattern configuration
USE_EDGE_DIAGONALS = True      # Whether to use edge diagonal patterns
EDGE_DIAGONAL_PROBABILITY = 0.4  # Probability of a rectangle having edge diagonal pattern
EDGE_DIAGONAL_LINE_WIDTH = 1     # Width of edge diagonal lines
EDGE_DIAGONAL_LENGTH = 8        # Length of edge diagonal lines extending inward
EDGE_DIAGONAL_SPACING = 8        # Spacing between edge diagonal lines
EDGE_DIAGONAL_COLOR = 'black'    # Color of edge diagonal lines

def calculate_structure_bounds(canvas_width, canvas_height, num_structures, structure_index):
    """
    Calculate the bounds for a structure based on its index and number of structures
    Returns: (left, right, top, bottom)
    """
    if num_structures == 1:
        return 0, canvas_width, 0, canvas_height
    
    elif num_structures == 2:
        # Split horizontally
        section_height = canvas_height // 2
        top = structure_index * section_height
        bottom = top + section_height
        return 0, canvas_width, top, bottom
    
    else:  # num_structures == 3
        if structure_index == 0:
            # First section (top half)
            return 0, canvas_width, 0, canvas_height // 2
        else:
            # Second and third sections (bottom half split vertically)
            section_width = canvas_width // 2
            left = (structure_index - 1) * section_width
            right = left + section_width
            return left, right, canvas_height // 2, canvas_height

def generate_rectangle_widths(total_width, num_rectangles):
    """
    Generate random widths for rectangles that sum up to total_width
    Ensures each rectangle has at least MIN_RECTANGLE_WIDTH
    Returns: list of widths
    """
    # First, reserve minimum width for each rectangle
    reserved_width = MIN_RECTANGLE_WIDTH * num_rectangles
    remaining_width = total_width - reserved_width
    
    if remaining_width < 0:
        raise ValueError(f"Total width {total_width} is too small for {num_rectangles} rectangles with minimum width {MIN_RECTANGLE_WIDTH}")
    
    # Generate random proportions for the remaining width
    proportions = [random.random() for _ in range(num_rectangles)]
    total_proportion = sum(proportions)
    
    # Scale proportions to match remaining width and add minimum width
    widths = [MIN_RECTANGLE_WIDTH + int((p / total_proportion) * remaining_width) for p in proportions]
    
    # Adjust for any rounding errors
    while sum(widths) != total_width:
        diff = total_width - sum(widths)
        if diff > 0:
            # Add to the largest rectangle
            largest_idx = widths.index(max(widths))
            widths[largest_idx] += 1
        else:
            # Subtract from the largest rectangle
            largest_idx = widths.index(max(widths))
            if widths[largest_idx] > MIN_RECTANGLE_WIDTH:
                widths[largest_idx] -= 1
    
    return widths

def draw_dotted_line(draw, start_point, end_point, width, color='black'):
    """
    Draw a dotted line between two points
    """
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Calculate line length and angle
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = math.atan2(y2 - y1, x2 - x1)
    
    # Calculate number of dots needed
    num_dots = int(length / (DOTTED_LINE_SPACING + DOTTED_LINE_LENGTH))
    
    # Draw dots along the line
    for i in range(num_dots):
        # Calculate dot position
        t = i / (num_dots - 1) if num_dots > 1 else 0.5
        dot_x = x1 + (x2 - x1) * t
        dot_y = y1 + (y2 - y1) * t
        
        # Calculate dot start and end points
        dot_start_x = dot_x - (DOTTED_LINE_LENGTH/2) * math.cos(angle)
        dot_start_y = dot_y - (DOTTED_LINE_LENGTH/2) * math.sin(angle)
        dot_end_x = dot_x + (DOTTED_LINE_LENGTH/2) * math.cos(angle)
        dot_end_y = dot_y + (DOTTED_LINE_LENGTH/2) * math.sin(angle)
        
        # Draw dot as a short line
        draw.line(
            [(dot_start_x, dot_start_y), (dot_end_x, dot_end_y)],
            fill=color,
            width=width
        )

def draw_center_line(draw, left, right, top, bottom):
    """
    Draw a center line in the given section
    Returns: (line_start_x, line_end_x, center_y)
    """
    section_width = right - left
    line_length = int(section_width * STRUCTURE_LENGTH_RATIO)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    # Calculate line start and end points
    line_start_x = center_x - (line_length // 2)
    line_end_x = center_x + (line_length // 2)
    draw.line([(line_start_x, center_y), (line_end_x, center_y)], fill='white', width=1)    
    return line_start_x, line_end_x, center_y

def draw_arrow(draw, x, y, direction='right', color='black', width=1, structure_width=None):
    """
    Draw a line-based arrow at the specified position
    direction: 'left' or 'right'
    Arrow is placed inside the horizontal line, pointing outward
    """
    # Calculate arrow dimensions based on structure width
    arrow_length = max(2, int(structure_width * ARROW_SIZE_RATIO))
    arrow_width = max(2, int(structure_width * ARROW_SIZE_RATIO))
    
    if direction == 'right':
        # Points for right-facing arrow (inside the line)
        arrow_x1 = x - arrow_length
        arrow_y1 = y - arrow_width
        arrow_x2 = x - arrow_length
        arrow_y2 = y + arrow_width
    else:  # left
        # Points for left-facing arrow (inside the line)
        arrow_x1 = x + arrow_length
        arrow_y1 = y - arrow_width
        arrow_x2 = x + arrow_length
        arrow_y2 = y + arrow_width
    
    # Draw arrow lines
    draw.line([(x, y), (arrow_x1, arrow_y1)], fill=color, width=width)
    draw.line([(x, y), (arrow_x2, arrow_y2)], fill=color, width=width)

def draw_mosaic(draw, x1, x2, y, height, line_spacing=MOSAIC_LINE_SPACING, line_width=MOSAIC_LINE_WIDTH):
    """
    Draw 45-degree and 135-degree diagonal lines within the rectangle,
    ensuring complete coverage including edges
    """
    half_height = height // 2
    top = y - half_height
    bottom = y + half_height
    width = x2 - x1
    
    # Calculate the total number of lines needed to cover the entire width
    # Add extra lines to ensure edge coverage
    num_lines = (width + height) // line_spacing + 2
    
    # Start from before the left edge to ensure coverage
    start_x = x1 - height
    
    # Draw 45-degree lines (bottom-left to top-right)
    for i in range(num_lines):
        # Calculate start point (bottom)
        current_x = start_x + (i * line_spacing)
        start_y = bottom
        
        # Calculate end point (45 degrees up and right)
        end_x = current_x + height  # For 45 degrees, rise equals run
        end_y = top
        
        # Draw the line if any part of it intersects with the rectangle
        if end_x >= x1 and current_x <= x2:
            # Clip the line to the rectangle bounds
            if current_x < x1:
                # Adjust start point to rectangle edge
                start_y = bottom - (x1 - current_x)
                current_x = x1
            if end_x > x2:
                # Adjust end point to rectangle edge
                end_y = top + (end_x - x2)
                end_x = x2
            
            draw.line(
                [(current_x, start_y), (end_x, end_y)],
                fill=MOSAIC_COLOR,
                width=line_width
            )
    
    # Draw 135-degree lines (bottom-right to top-left)
    for i in range(num_lines):
        # Calculate start point (bottom)
        current_x = x2 + height - (i * line_spacing)
        start_y = bottom
        
        # Calculate end point (135 degrees up and left)
        end_x = current_x - height  # For 135 degrees, rise equals run
        end_y = top
        
        # Draw the line if any part of it intersects with the rectangle
        if end_x <= x2 and current_x >= x1:
            # Clip the line to the rectangle bounds
            if current_x > x2:
                # Adjust start point to rectangle edge
                start_y = bottom - (current_x - x2)
                current_x = x2
            if end_x < x1:
                # Adjust end point to rectangle edge
                end_y = top + (x1 - end_x)
                end_x = x1
            
            draw.line(
                [(current_x, start_y), (end_x, end_y)],
                fill=MOSAIC_COLOR,
                width=line_width
            )

def draw_diagonal(draw, x1, x2, y, height):
    """
    Draw a diagonal line from top-left to bottom-right within the rectangle
    Returns: angle of the diagonal line in degrees
    """
    half_height = height // 2
    top = y - half_height
    bottom = y + half_height
    
    # Calculate gaps with line thickness consideration
    left_gap = LINE_GAP + HORIZONTAL_LINE_WIDTH
    right_gap = LINE_GAP
    
    # Calculate start and end points
    start_x = x1 + left_gap
    start_y = top
    end_x = x2 - right_gap
    end_y = bottom
    
    # Draw diagonal line
    if USE_DOTTED_LINES and random.random() < DOTTED_LINE_PROBABILITY:
        draw_dotted_line(draw, (start_x, start_y), (end_x, end_y), DIAGONAL_LINE_WIDTH)
    else:
        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=DIAGONAL_COLOR,
            width=DIAGONAL_LINE_WIDTH
        )
    
    # Calculate the angle of the diagonal line
    dx = end_x - start_x
    dy = end_y - start_y
    angle = math.degrees(math.atan2(dy, dx))
    
    # Mirror the angle vertically and flip text right-side up
    mirrored_angle = 180 - angle + 180
    
    return mirrored_angle, (start_x, start_y, end_x, end_y)

def draw_edge_diagonals(draw, x1, x2, y, height):
    """
    Draw diagonal lines along the top and right edges of the rectangle, extending inward at 45 degrees
    """
    half_height = height // 2
    top = y - half_height
    bottom = y + half_height
    
    # Calculate gaps with line thickness consideration
    left_gap = LINE_GAP + HORIZONTAL_LINE_WIDTH
    right_gap = LINE_GAP
    
    # Calculate effective rectangle boundaries
    rect_left = x1 + left_gap
    rect_right = x2 - right_gap
    rect_top = top
    rect_bottom = bottom
    
    # Draw diagonal lines along the top edge
    # Calculate number of lines that can fit along the top edge
    top_width = rect_right - rect_left
    num_top_lines = max(1, int(top_width / EDGE_DIAGONAL_SPACING))
    
    for i in range(num_top_lines):
        # Calculate starting point along the top edge
        start_x = rect_left + (i * EDGE_DIAGONAL_SPACING)
        if start_x >= rect_right:
            break
            
        start_y = rect_top
        
        # Calculate end point extending downward at 45 degrees
        end_x = start_x + EDGE_DIAGONAL_LENGTH
        end_y = start_y + EDGE_DIAGONAL_LENGTH
        
        # Clip the line to stay within the rectangle bounds
        if end_x > rect_right:
            # Adjust end point to stay within right boundary
            end_x = rect_right
            end_y = start_y + (end_x - start_x)  # Maintain 45-degree angle
        
        if end_y > rect_bottom:
            # Adjust end point to stay within bottom boundary
            end_y = rect_bottom
            end_x = start_x + (end_y - start_y)  # Maintain 45-degree angle
        
        # Draw the diagonal line (always solid, never dotted)
        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=EDGE_DIAGONAL_COLOR,
            width=EDGE_DIAGONAL_LINE_WIDTH
        )
    
    # Draw diagonal lines along the right edge
    # Calculate number of lines that can fit along the right edge
    right_height = rect_bottom - rect_top
    num_right_lines = max(1, int(right_height / EDGE_DIAGONAL_SPACING))
    
    for i in range(num_right_lines):
        # Calculate starting point along the right edge
        start_x = rect_right
        start_y = rect_top + (i * EDGE_DIAGONAL_SPACING)
        if start_y >= rect_bottom:
            break
        
        # Calculate end point extending leftward at 45 degrees
        end_x = start_x - EDGE_DIAGONAL_LENGTH
        end_y = start_y + EDGE_DIAGONAL_LENGTH
        
        # Clip the line to stay within the rectangle bounds
        if end_x < rect_left:
            # Adjust end point to stay within left boundary
            end_x = rect_left
            end_y = start_y + (start_x - end_x)  # Maintain 45-degree angle
        
        if end_y > rect_bottom:
            # Adjust end point to stay within bottom boundary
            end_y = rect_bottom
            end_x = start_x - (end_y - start_y)  # Maintain 45-degree angle
        
        # Draw the diagonal line (always solid, never dotted)
        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=EDGE_DIAGONAL_COLOR,
            width=EDGE_DIAGONAL_LINE_WIDTH
        )

def generate_single_text_line():
    """
    Generate a single line of text with specified number and letter ratio
    Returns: single line of text
    """
    # Randomly choose text length
    text_length = random.randint(MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
    
    # Calculate number of digits and letters
    num_digits = int(text_length * NUMBER_RATIO)
    num_letters = text_length - num_digits
    
    # Generate numbers and letters
    numbers = ''.join(random.choices('0123456789', k=num_digits))
    letters = ''.join(random.choices('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz', k=num_letters))
    
    # Combine and shuffle
    combined = list(numbers + letters)
    random.shuffle(combined)
    text = ''.join(combined)
    
    # Add special symbol if enabled and probability check passes
    if USE_SPECIAL_SYMBOLS and random.random() < SPECIAL_SYMBOL_PROBABILITY:
        symbol = random.choice(ADDITIONAL_SYMBOLS)
        # Split text in half and insert symbol in the middle
        mid_point = len(text) // 2
        text = text[:mid_point] + symbol + text[mid_point:]
    
    return text

def generate_text():
    """
    Generate text with specified number and letter ratio, potentially splitting some rows
    Returns: tuple of (vertical_text, horizontal_text_segments)
    vertical_text: single line of text for vertical placement
    horizontal_text_segments: list of (text, row_index, segment_index) for horizontal placement
    """
    # Generate vertical text (always single line, no splitting for vertical text)
    vertical_text = generate_single_text_line()
    
    # Generate horizontal text (can be multiple rows, potentially split)
    num_rows = random.randint(MIN_TEXT_ROWS, MAX_TEXT_ROWS)
    horizontal_text_segments = []  # Each element is (text, row_index, segment_index)
    
    for row_idx in range(num_rows):
        base_text = generate_single_text_line()
        
        # Decide if this row should be split into 2 segments
        if random.random() < TEXT_SPLIT_PROBABILITY and len(base_text) >= 4:  # Only split if text is long enough
            # Split the text roughly in the middle
            split_point = len(base_text) // 2 + random.randint(-1, 1)  # Add some randomness
            split_point = max(1, min(len(base_text) - 1, split_point))  # Ensure valid split
            
            segment1 = base_text[:split_point]
            segment2 = base_text[split_point:]
            
            horizontal_text_segments.append((segment1, row_idx, 0))  # First segment of the row
            horizontal_text_segments.append((segment2, row_idx, 1))  # Second segment of the row
        else:
            # Keep as single segment
            horizontal_text_segments.append((base_text, row_idx, 0))
    
    return vertical_text, horizontal_text_segments

def get_random_font():
    if FONTS[0] == "default":
        return ImageFont.load_default()
        
    font_name = random.choice(FONTS)
    font_path = os.path.join(FONT_DIR, font_name)
    try:
        return ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        print(f"Warning: Could not load font {font_name}: {str(e)}")
        return ImageFont.load_default()

def calculate_overlap(box1, box2):
    """
    Calculate the overlap percentage between two text boxes
    box format: (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
    Returns: overlap percentage (0 to 1)
    """
    # Calculate intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate overlap percentage relative to the smaller box
    smaller_area = min(box1_area, box2_area)
    if smaller_area == 0:
        return 0.0
    
    return intersection_area / smaller_area

def resolve_text_overlaps(text_boxes):
    """
    Resolve overlapping texts based on overlap tolerance
    text_boxes: list of tuples (box, text_type) where box is (x1,y1,x2,y2)
    Returns: list of indices of texts to keep
    """
    if not text_boxes:
        return []
    
    # Find all overlaps
    overlaps = []
    print("\nText Overlap Analysis:")
    print("-" * 50)
    for i in range(len(text_boxes)):
        for j in range(i + 1, len(text_boxes)):
            overlap = calculate_overlap(text_boxes[i][0], text_boxes[j][0])
            print(f"Text {i} ({text_boxes[i][1]}) vs Text {j} ({text_boxes[j][1]})")
            print(f"Box {i}: {text_boxes[i][0]}")
            print(f"Box {j}: {text_boxes[j][0]}")
            print(f"Overlap: {overlap:.2%}")
            print("-" * 50)
            if overlap > TEXT_OVERLAP_TOLERANCE:
                overlaps.append((i, j, overlap))
    
    # If no overlaps, keep all texts
    if not overlaps:
        print("No texts exceeded overlap tolerance. Keeping all texts.")
        return list(range(len(text_boxes)))
    
    # Resolve overlaps
    texts_to_keep = set(range(len(text_boxes)))
    print("\nResolving overlaps:")
    for i, j, overlap in overlaps:
        if i in texts_to_keep and j in texts_to_keep:
            # Randomly choose one to keep
            to_remove = random.choice([i, j])
            texts_to_keep.remove(to_remove)
            print(f"Overlap between Text {i} ({text_boxes[i][1]}) and Text {j} ({text_boxes[j][1]})")
            print(f"Overlap: {overlap:.2%}")
            print(f"Removing Text {to_remove} ({text_boxes[to_remove][1]})")
            print("-" * 50)
    
    print(f"\nFinal texts to keep: {sorted(texts_to_keep)}")
    return list(texts_to_keep)

def draw_strikethrough(draw, text_box, text_lines, font, is_diagonal=False, is_vertical=False):
    """
    Draw a strikethrough line through the text box
    text_box: (x1, y1, x2, y2) coordinates of the text box
    text_lines: list of text lines or single text string to strikethrough
    font: the font used for the text
    is_diagonal: whether to draw diagonal strikethrough
    is_vertical: whether to draw vertical strikethrough
    """
    x1, y1, x2, y2 = text_box
    
    # Handle both single text and multiple text lines
    if isinstance(text_lines, str):
        text_lines = [text_lines]
    
    if STRIKETHROUGH_WORD_BASED:
        # Calculate line height for multi-row text
        line_height = FONT_SIZE + TEXT_ROW_SPACING
        current_y = y1
        
        for line_idx, text_line in enumerate(text_lines):
            # Split text into words and calculate positions
            words = text_line.split()
            current_x = x1
            line_y2 = current_y + FONT_SIZE
            
            for word in words:
                word_width = font.getlength(word)
                word_box = (current_x, current_y, current_x + word_width, line_y2)
                
                if is_diagonal:
                    # Draw diagonal strikethrough for word
                    draw.line(
                        [(word_box[0], word_box[1]), (word_box[2], word_box[3])],
                        fill=STRIKETHROUGH_COLOR,
                        width=STRIKETHROUGH_WIDTH
                    )
                elif is_vertical:
                    # Draw vertical strikethrough for word
                    center_x = (word_box[0] + word_box[2]) // 2
                    draw.line(
                        [(center_x, word_box[1]), (center_x, word_box[3])],
                        fill=STRIKETHROUGH_COLOR,
                        width=STRIKETHROUGH_WIDTH
                    )
                else:
                    # Draw horizontal strikethrough for word
                    center_y = (word_box[1] + word_box[3]) // 2
                    draw.line(
                        [(word_box[0], center_y), (word_box[2], center_y)],
                        fill=STRIKETHROUGH_COLOR,
                        width=STRIKETHROUGH_WIDTH
                    )
                
                # Add space width for next word
                current_x += word_width + font.getlength(' ')
            
            # Move to next line
            current_y += line_height
    else:
        # Draw strikethrough for entire text block
        if is_diagonal:
            draw.line(
                [(x1, y1), (x2, y2)],
                fill=STRIKETHROUGH_COLOR,
                width=STRIKETHROUGH_WIDTH
            )
        elif is_vertical:
            center_x = (x1 + x2) // 2
            draw.line(
                [(center_x, y1), (center_x, y2)],
                fill=STRIKETHROUGH_COLOR,
                width=STRIKETHROUGH_WIDTH
            )
        else:
            center_y = (y1 + y2) // 2
            draw.line(
                [(x1, center_y), (x2, center_y)],
                fill=STRIKETHROUGH_COLOR,
                width=STRIKETHROUGH_WIDTH
            )

def draw_tolerance_box(draw, x1, x2, y_top, box_width, left_word, right_word, image):
    """
    Draw a tolerance box above the rectangle from x1 to x2, with two adjacent boxes:
    left box (single word), right box (longer word).
    y_top: the y coordinate of the bottom of the tolerance box (i.e., top of the main rectangle minus spacing)
    box_width: width of the main rectangle
    left_word: text for the left box (will be ignored, using special character instead)
    right_word: text for the right box
    image: PIL image for pasting text if needed
    Returns: list of bounding boxes for the text
    """
    # Use seguisym.ttf for left word (special character)
    try:
        special_font = ImageFont.truetype('fonts/seguisym.ttf', TOLERANCE_BOX_FONT_SIZE)
    except:
        print("Warning: Could not load seguisym.ttf, using default font")
        special_font = ImageFont.load_default()
    
    # Use regular font for right word
    font = get_random_font()
    if hasattr(font, 'font') and hasattr(font.font, 'size'):
        font = ImageFont.truetype(font.path, TOLERANCE_BOX_FONT_SIZE) if hasattr(font, 'path') else font
    else:
        font = ImageFont.load_default()

    # Use special character for left word
    left_word = random.choice(special_char_segu)

    # Calculate text sizes
    left_text_width = special_font.getlength(left_word)
    right_text_width = font.getlength(right_word)

    # Box widths (text + padding)
    left_box_width = int(left_text_width + 2 * TOLERANCE_BOX_TEXT_PADDING)
    right_box_width = int(right_text_width + 2 * TOLERANCE_BOX_TEXT_PADDING)

    # Calculate box height based on font size and padding
    box_height = TOLERANCE_BOX_FONT_SIZE + 2 * TOLERANCE_BOX_TEXT_PADDING

    # Center the tolerance box horizontally above the main rectangle
    total_box_width = left_box_width + right_box_width
    center_x = (x1 + x2) // 2
    box_x1 = center_x - total_box_width // 2
    box_x2 = box_x1 + total_box_width

    # Calculate random spacing between tolerance box and main rectangle
    spacing = random.randint(TOLERANCE_BOX_MIN_SPACING, TOLERANCE_BOX_MAX_SPACING)
    box_y2 = y_top - spacing
    box_y1 = box_y2 - box_height

    # Draw left box
    draw.rectangle([(box_x1, box_y1), (box_x1 + left_box_width, box_y2)], outline='black', width=TOLERANCE_BOX_LINE_WIDTH)
    # Draw right box
    draw.rectangle([(box_x1 + left_box_width, box_y1), (box_x2, box_y2)], outline='black', width=TOLERANCE_BOX_LINE_WIDTH)

    # Calculate text positions
    left_text_x = box_x1 + (left_box_width - left_text_width) // 2
    left_text_y = box_y1 + (box_height - TOLERANCE_BOX_FONT_SIZE + 10) // 2 #-10 to account for font file error
    right_text_x = box_x1 + left_box_width + (right_box_width - right_text_width) // 2
    right_text_y = box_y1 + (box_height - TOLERANCE_BOX_FONT_SIZE + 10) // 2 #-10 to account for font file error

    # Draw left text (centered) with special font
    draw.text((left_text_x, left_text_y-10), left_word, fill='black', font=special_font) #-10 to account for font file error

    # Draw right text (centered)
    draw.text((right_text_x, right_text_y), right_word, fill='black', font=font)

    # Draw connecting line from left edge (center vertically) of tolerance box
    # 1. Start at left edge, center vertically
    start_x = box_x1
    start_y = (box_y1 + box_y2) // 2
    # 2. Extend 10 pixels left
    mid_x = start_x - 10
    mid_y = start_y
    # 3. Go vertically down to the top edge of the main rectangle (y_top)
    end_x = mid_x
    end_y = y_top
    # Draw horizontal segment
    draw.line([(start_x, start_y), (mid_x, mid_y)], fill='black', width=TOLERANCE_BOX_LINE_WIDTH)
    # Draw vertical segment
    draw.line([(mid_x, mid_y), (end_x, end_y)], fill='black', width=TOLERANCE_BOX_LINE_WIDTH)

    # Create bounding boxes for the text with padding
    bounding_boxes = []
    
    # Only add left text box if not disabled
    if not DISABLE_LEFT_TOLERANCE_BOX_BOUNDING:
        left_text_box = (left_text_x, left_text_y, 
                        left_text_x + left_text_width, 
                        left_text_y + TOLERANCE_BOX_FONT_SIZE-10)
        left_text_box_with_padding = add_padding_to_box(left_text_box, BOUNDING_BOX_PADDING)
        bounding_boxes.append(left_text_box_with_padding)
    
    # Always add right text box
    right_text_box = (right_text_x, right_text_y, 
                     right_text_x + right_text_width, 
                     right_text_y + TOLERANCE_BOX_FONT_SIZE-10) #-10 to account for font file error
    right_text_box_with_padding = add_padding_to_box(right_text_box, BOUNDING_BOX_PADDING)
    bounding_boxes.append(right_text_box_with_padding)

    return bounding_boxes

def add_padding_to_box(box, padding):
    """
    Add padding to a bounding box
    box: (x1, y1, x2, y2)
    padding: amount of padding to add
    Returns: (x1, y1, x2, y2) with padding
    """
    x1, y1, x2, y2 = box
    return (x1 - padding, y1 - padding, x2 + padding, y2 + padding)

def process_bounding_boxes_for_format(bounding_boxes):
    """
    Process bounding boxes into the exact format that will be saved to .txt files.
    Returns a list of processed boxes in the format that matches the output format.
    
    For YOLO format: returns (class_id, x_center, y_center, width, height)
    For polygon format: returns (x1, y1, x2, y2, x3, y3, x4, y4) as integers
    """
    processed_boxes = []
    
    for box in bounding_boxes:
        orig_x1, orig_y1, orig_x2, orig_y2 = box
        # Ensure coordinates are properly ordered
        orig_x1, orig_x2 = min(orig_x1, orig_x2), max(orig_x1, orig_x2)
        orig_y1, orig_y2 = min(orig_y1, orig_y2), max(orig_y1, orig_y2)
        
        if USE_YOLO_FORMAT:
            # Convert to YOLO format (normalized coordinates)
            img_width = CANVAS_WIDTH
            img_height = CANVAS_HEIGHT
            
            # Calculate center, width, and height
            x_center = (orig_x1 + orig_x2) / 2 / img_width
            y_center = (orig_y1 + orig_y2) / 2 / img_height
            width = (orig_x2 - orig_x1) / img_width
            height = (orig_y2 - orig_y1) / img_height
            
            processed_boxes.append((YOLO_CLASS_ID, x_center, y_center, width, height))
        else:
            # Convert to 4-point polygon format (clockwise from top-left)
            # Use original coordinates to avoid variable collision
            x1 = round(orig_x1)  # top-left x
            y1 = round(orig_y1)  # top-left y
            x2 = round(orig_x2)  # top-right x  
            y2 = round(orig_y1)  # top-right y (same as top-left y)
            x3 = round(orig_x2)  # bottom-right x
            y3 = round(orig_y2)  # bottom-right y
            x4 = round(orig_x1)  # bottom-left x
            y4 = round(orig_y2)  # bottom-left y (same as bottom-right y)
            
            processed_boxes.append((x1, y1, x2, y2, x3, y3, x4, y4))
    
    return processed_boxes

def draw_rectangle(draw, x1, x2, y, height, structure_width, image, prev_rect_texts=None):
    half_height = height // 2
    
    # Calculate points
    top_left = (x1, y - half_height)
    top_right = (x2, y - half_height)
    bottom_right = (x2, y + half_height)
    bottom_left = (x1, y + half_height)
    
    # Calculate gaps with line thickness consideration
    left_gap = LINE_GAP + HORIZONTAL_LINE_WIDTH
    right_gap = LINE_GAP
    
    # Draw horizontal lines with gaps
    if USE_DOTTED_LINES and random.random() < DOTTED_LINE_PROBABILITY:
        draw_dotted_line(draw, (x1 + left_gap, y - half_height), (x2 - right_gap, y - half_height), 
                        HORIZONTAL_LINE_WIDTH)  # Top line
    else:
        draw.line([(x1 + left_gap, y - half_height), (x2 - right_gap, y - half_height)], 
                 fill='black', width=HORIZONTAL_LINE_WIDTH)  # Top line
    
    if USE_DOTTED_LINES and random.random() < DOTTED_LINE_PROBABILITY:
        draw_dotted_line(draw, (x1 + left_gap, y + half_height), (x2 - right_gap, y + half_height), 
                        HORIZONTAL_LINE_WIDTH)  # Bottom line
    else:
        draw.line([(x1 + left_gap, y + half_height), (x2 - right_gap, y + half_height)], 
                 fill='black', width=HORIZONTAL_LINE_WIDTH)  # Bottom line
    
    # Draw vertical lines
    if USE_DOTTED_LINES and random.random() < DOTTED_LINE_PROBABILITY:
        draw_dotted_line(draw, top_left, bottom_left, VERTICAL_LINE_WIDTH)
    else:
        draw.line([top_left, bottom_left], fill='black', width=VERTICAL_LINE_WIDTH)
    
    if USE_DOTTED_LINES and random.random() < DOTTED_LINE_PROBABILITY:
        draw_dotted_line(draw, top_right, bottom_right, VERTICAL_LINE_WIDTH)
    else:
        draw.line([top_right, bottom_right], fill='black', width=VERTICAL_LINE_WIDTH)
    
    # Draw arrows at the ends of horizontal lines
    # Top line arrows (pointing outward)
    draw_arrow(draw, x1 + left_gap, y - half_height, direction='left', 
              width=HORIZONTAL_LINE_WIDTH, structure_width=structure_width)  # Left arrow
    draw_arrow(draw, x2 - right_gap, y - half_height, direction='right', 
              width=HORIZONTAL_LINE_WIDTH, structure_width=structure_width)  # Right arrow
    
    # Bottom line arrows (pointing outward)
    draw_arrow(draw, x1 + left_gap, y + half_height, direction='left', 
              width=HORIZONTAL_LINE_WIDTH, structure_width=structure_width)  # Left arrow
    draw_arrow(draw, x2 - right_gap, y + half_height, direction='right', 
              width=HORIZONTAL_LINE_WIDTH, structure_width=structure_width)  # Right arrow
    
    # Store all text boxes for overlap detection
    text_boxes = []
    bounding_boxes = []  # Store bounding boxes with padding
    
    # Generate and prepare all texts first
    left_text, bottom_text_segments = generate_text()
    font = get_random_font()
    
    # Calculate text dimensions
    left_text_width = font.getlength(left_text)
    text_height = FONT_SIZE + 1
    
    # Calculate dimensions for multi-row horizontal text
    max_bottom_text_width = max(font.getlength(segment[0]) for segment in bottom_text_segments)
    
    # Calculate total height based on number of rows (not segments)
    num_rows = max(segment[1] for segment in bottom_text_segments) + 1 if bottom_text_segments else 0
    total_bottom_text_height = num_rows * FONT_SIZE + (num_rows - 1) * TEXT_ROW_SPACING
    
    # Calculate available space for text
    available_width = x2 - right_gap - (x1 + left_gap)
    available_height = height
    
    # Handle pattern selection and drawing (mutually exclusive patterns)
    has_diagonal = False
    diagonal_angle = 0
    diagonal_coords = None
    
    # Create list of available patterns
    available_patterns = []
    if USE_MOSAIC:
        available_patterns.append(('mosaic', MOSAIC_PROBABILITY))
    if USE_DIAGONAL:
        available_patterns.append(('diagonal', DIAGONAL_PROBABILITY))
    if USE_EDGE_DIAGONALS:
        available_patterns.append(('edge_diagonal', EDGE_DIAGONAL_PROBABILITY))
    
    # Randomly select one pattern based on probabilities
    if available_patterns:
        # Choose pattern based on weighted random selection
        for pattern_name, probability in available_patterns:
            if random.random() < probability:
                if pattern_name == 'mosaic':
                    mosaic_height = height - (2 * MOSAIC_GAP)
                    draw_mosaic(draw, x1 + left_gap, x2 - right_gap, y, mosaic_height)
                    break
                elif pattern_name == 'diagonal':
                    diagonal_angle, diagonal_coords = draw_diagonal(draw, x1, x2, y, height)
                    has_diagonal = True
                    break
                elif pattern_name == 'edge_diagonal':
                    draw_edge_diagonals(draw, x1, x2, y, height)
                    break
    
    # Create and rotate left text
    text_image = Image.new('RGBA', (int(left_text_width), text_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((0, 0), left_text, fill='black', font=font)
    
    # Rotate the text image 90 degrees counterclockwise
    rotated_text_image = text_image.rotate(90, expand=True)
    
    # Calculate position to paste rotated text
    rotated_width, rotated_height = rotated_text_image.size
    paste_x = int(x1 - VERTICAL_TEXT_SPACING - rotated_width)
    
    # Position vertical text at bottom-left if enabled, otherwise center
    if VERTICAL_TEXT_BOTTOM_LEFT:
        paste_y = int(y + half_height - rotated_height)
    else:
        paste_y = int(y - rotated_height // 2)
    
    # Add left text box with padding
    left_box = (paste_x, paste_y, paste_x + rotated_width, paste_y + rotated_height)
    left_box_with_padding = add_padding_to_box(left_box, BOUNDING_BOX_PADDING)
    text_boxes.append((left_box, 'left'))
    bounding_boxes.append(left_box_with_padding)
    
    # Calculate bottom text position for multi-row text
    if HORIZONTAL_TEXT_BOTTOM_LEFT:
        # Position at bottom left with spacing
        text_x = x1 + left_gap
    else:
        # Center the text based on the widest line
        text_x = x1 + left_gap + ((available_width - max_bottom_text_width) // 2)
    
    # Position the text block so the bottom line is at the correct position
    text_y = y + half_height - total_bottom_text_height - HORIZONTAL_TEXT_SPACING
    
    # Create individual bounding boxes for each text segment
    for segment_text, row_idx, segment_idx in bottom_text_segments:
        # Calculate position for this segment
        segment_y = text_y + row_idx * (FONT_SIZE + TEXT_ROW_SPACING)
        
        # Calculate x position for this segment
        if segment_idx == 0:
            # First segment of the row
            segment_x = text_x
        else:
            # Second segment - find first segment width and add spacing
            first_segment = None
            for other_text, other_row, other_seg in bottom_text_segments:
                if other_row == row_idx and other_seg == 0:
                    first_segment = other_text
                    break
            if first_segment:
                first_width = font.getlength(first_segment)
                segment_x = text_x + first_width + TEXT_SEGMENT_SPACING
            else:
                segment_x = text_x
        
        # Calculate segment dimensions
        segment_width = font.getlength(segment_text)
        segment_height = FONT_SIZE
        
        # Create bounding box for this segment
        segment_box = (segment_x, segment_y, segment_x + segment_width, segment_y + segment_height)
        segment_box_with_padding = add_padding_to_box(segment_box, BOUNDING_BOX_PADDING)
        text_boxes.append((segment_box, f'bottom_r{row_idx}_s{segment_idx}'))
        bounding_boxes.append(segment_box_with_padding)
    
    # Handle diagonal text if present
    if has_diagonal and diagonal_coords:
        diagonal_text = generate_single_text_line()  # Diagonal text is always single line
        diagonal_font = get_random_font()
        
        # Calculate diagonal text dimensions
        diagonal_text_width = diagonal_font.getlength(diagonal_text)
        
        # Create and rotate diagonal text
        diagonal_text_image = Image.new('RGBA', (int(diagonal_text_width), text_height), (255, 255, 255, 0))
        diagonal_text_draw = ImageDraw.Draw(diagonal_text_image)
        diagonal_text_draw.text((0, 0), diagonal_text, fill='black', font=diagonal_font)
        
        # Rotate the text image to match the diagonal line angle
        rotated_diagonal_text = diagonal_text_image.rotate(diagonal_angle, expand=True)
        
        # Calculate position to paste rotated diagonal text
        rotated_diag_width, rotated_diag_height = rotated_diagonal_text.size
        
        # Calculate center point of the diagonal line
        start_x, start_y, end_x, end_y = diagonal_coords
        center_x = (start_x + end_x) // 2
        center_y = (start_y + end_y) // 2
        
        # Calculate the perpendicular offset considering both spacing and font size
        angle_rad = math.radians(diagonal_angle)
        perp_angle_rad = angle_rad + math.pi/2
        # Add font size to spacing to ensure text doesn't overlap with line
        total_spacing = DIAGONAL_TEXT_SPACING + FONT_SIZE
        offset_x = math.cos(perp_angle_rad) * total_spacing
        offset_y = math.sin(perp_angle_rad) * total_spacing
        
        # Position text with offset from the diagonal line
        diag_paste_x = int(center_x - rotated_diag_width // 2 + offset_x )
        diag_paste_y = int(center_y - rotated_diag_height // 2 + offset_y )
        
        # Add diagonal text box with padding
        diag_box = (diag_paste_x, diag_paste_y, 
                   diag_paste_x + rotated_diag_width, 
                   diag_paste_y + rotated_diag_height)
        diag_box_with_padding = add_padding_to_box(diag_box, BOUNDING_BOX_PADDING)
        text_boxes.append((diag_box, 'diagonal'))
        bounding_boxes.append(diag_box_with_padding)
    
    # Draw tolerance box if enabled and rectangle is wide enough
    tolerance_boxes = []
    if USE_TOLERANCE_BOX and (x2 - x1) >= TOLERANCE_BOX_MIN_WIDTH:
        left_word = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))
        right_word = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(3, 6)))
        tolerance_boxes = draw_tolerance_box(
            draw, x1, x2, y - half_height, x2 - x1, left_word, right_word, image
        )
        bounding_boxes.extend(tolerance_boxes)
    
    # Combine current text boxes with previous rectangle's text boxes for overlap checking
    all_text_boxes = text_boxes.copy()
    if prev_rect_texts:
        all_text_boxes.extend(prev_rect_texts)
    
    # Resolve text overlaps
    texts_to_keep = resolve_text_overlaps(all_text_boxes)
    
    # Filter texts_to_keep to only include current rectangle's texts
    current_texts_to_keep = [i for i in texts_to_keep if i < len(text_boxes)]
    
    # Create new lists for kept texts and bounding boxes
    kept_text_boxes = []
    kept_bounding_boxes = []
    
    # Draw the texts that were kept and collect their bounding boxes
    for i in current_texts_to_keep:
        box, text_type = text_boxes[i]
        kept_text_boxes.append((box, text_type))
        kept_bounding_boxes.append(bounding_boxes[i])
        
        if text_type == 'left':
            image.paste(rotated_text_image, (box[0], box[1]), rotated_text_image)
            if USE_TEXT_STRIKETHROUGH and random.random() < STRIKETHROUGH_PROBABILITY:
                rand = random.random()
                is_diagonal = rand < STRIKETHROUGH_DIAGONAL_PROBABILITY
                is_vertical = rand < STRIKETHROUGH_VERTICAL_PROBABILITY and not is_diagonal
                draw_strikethrough(draw, box, left_text, font, is_diagonal, is_vertical)
        elif text_type.startswith('bottom_'):
            # Draw individual text segment
            # Extract the segment info from the text_type
            parts = text_type.split('_')
            row_idx = int(parts[1][1:])  # Extract number from 'r#'
            segment_idx = int(parts[2][1:])  # Extract number from 's#'
            
            # Find the corresponding segment text
            segment_text = None
            for seg_text, seg_row, seg_seg in bottom_text_segments:
                if seg_row == row_idx and seg_seg == segment_idx:
                    segment_text = seg_text
                    break
            
            if segment_text:
                draw.text((box[0], box[1]), segment_text, fill='black', font=font)
                
                if USE_TEXT_STRIKETHROUGH and random.random() < STRIKETHROUGH_PROBABILITY:
                    rand = random.random()
                    is_diagonal = rand < STRIKETHROUGH_DIAGONAL_PROBABILITY
                    is_vertical = rand < STRIKETHROUGH_VERTICAL_PROBABILITY and not is_diagonal
                    draw_strikethrough(draw, box, segment_text, font, is_diagonal, is_vertical)
        elif text_type == 'diagonal':
            image.paste(rotated_diagonal_text, (box[0], box[1]), rotated_diagonal_text)
            if USE_TEXT_STRIKETHROUGH and random.random() < STRIKETHROUGH_PROBABILITY:
                rand = random.random()
                is_diagonal = rand < STRIKETHROUGH_DIAGONAL_PROBABILITY
                is_vertical = rand < STRIKETHROUGH_VERTICAL_PROBABILITY and not is_diagonal
                draw_strikethrough(draw, box, diagonal_text, font, is_diagonal, is_vertical)
    
    # Add tolerance boxes to kept bounding boxes if they exist
    kept_bounding_boxes.extend(tolerance_boxes)
    
    return kept_text_boxes, kept_bounding_boxes

def generate_structure(draw, left, right, top, bottom, structure_index, total_structures, image):
    """
    Generate rectangles for a single structure
    structure_index: index of the current structure (0-based)
    total_structures: total number of structures being generated
    image: the main image to paste rotated text onto
    Returns: list of bounding boxes
    """
    # Draw center line and get its bounds
    line_start_x, line_end_x, center_y = draw_center_line(draw, left, right, top, bottom)
    line_length = line_end_x - line_start_x
    
    # Calculate structure width for arrow sizing
    structure_width = right - left
    
    # Randomly choose number of rectangles between MIN_RECTANGLES and MAX_RECTANGLES
    # Use structure_index to ensure different numbers for each structure
    random.seed(structure_index)  # Set seed based on structure index
    num_rectangles = random.randint(MIN_RECTANGLES, MAX_RECTANGLES)
    random.seed()  # Reset seed to use system time
    
    # Generate rectangle widths
    rectangle_widths = generate_rectangle_widths(line_length, num_rectangles)
    
    # Draw rectangles
    current_x = line_start_x
    prev_rect_texts = None
    all_bounding_boxes = []
    rect_bottoms = []  # To store bottom y and x positions of rectangles
    for idx, width in enumerate(rectangle_widths):
        # Calculate rectangle end position
        next_x = current_x + width
        
        # Generate random height
        height = int((bottom - top) * random.uniform(MIN_HEIGHT_RATIO, MAX_HEIGHT_RATIO))
        
        # Draw rectangle with structure width for arrow sizing
        prev_rect_texts, rect_bounding_boxes = draw_rectangle(draw, current_x, next_x, center_y, height, structure_width, image, prev_rect_texts)
        all_bounding_boxes.extend(rect_bounding_boxes)
        
        # Store bottom corners for first and last rectangle
        half_height = height // 2
        if idx == 0:
            first_rect_bottom_left = (current_x, center_y + half_height)
        if idx == len(rectangle_widths) - 1:
            last_rect_bottom_right = (next_x, center_y + half_height)
        
        # Update current_x for next rectangle
        current_x = next_x
    
    # Draw the vertical and horizontal lines with text if there are rectangles
    if num_rectangles > 0:
        # Calculate y_target (10% from the bottom of the structure)
        y_target = int(bottom - 0.05 * (bottom - top))
        
        # Draw vertical lines
        draw.line([first_rect_bottom_left, (first_rect_bottom_left[0], y_target)], fill='black', width=1)
        draw.line([last_rect_bottom_right, (last_rect_bottom_right[0], y_target)], fill='black', width=1)
        
        # Generate and prepare text first to get its dimensions
        text = generate_single_text_line()  # Bottom structure text is always single line
        font = get_random_font()
        text_width = font.getlength(text)
        text_height = FONT_SIZE
        
        # Calculate text position (centered on the line)
        mid_x = (first_rect_bottom_left[0] + last_rect_bottom_right[0]) // 2
        text_x = mid_x - text_width // 2
        text_y = y_target - text_height // 2 - 2  # -2 to account for line width
        
        # Draw horizontal line segments with gap for text
        left_line_end = text_x - 10  # 10 pixels padding before text
        right_line_start = text_x + text_width + 10  # 10 pixels padding after text
        
        # Draw left segment of horizontal line
        draw.line([(first_rect_bottom_left[0], y_target-5), (left_line_end, y_target-5)], 
                 fill='black', width=1)
        
        # Draw right segment of horizontal line
        draw.line([(right_line_start, y_target-5), (last_rect_bottom_right[0], y_target-5)], 
                 fill='black', width=1)
        
        # Draw arrows
        draw_arrow(draw, int(first_rect_bottom_left[0]), y_target-5, direction='left', 
                  width=1, structure_width=800)  # Left arrow
        draw_arrow(draw, int(last_rect_bottom_right[0]), y_target-5, direction='right', 
                  width=1, structure_width=800)  # Right arrow
        
        # Draw text centered on the line
        draw.text((text_x, text_y), text, fill='black', font=font)
        
        # Add bounding box for the text
        text_box = (text_x, text_y, text_x + text_width, text_y + text_height)
        text_box_with_padding = add_padding_to_box(text_box, BOUNDING_BOX_PADDING)
        all_bounding_boxes.append(text_box_with_padding)
    
    return all_bounding_boxes

def save_bounding_boxes(bounding_boxes, image_num, is_test=False):
    """
    Save bounding box coordinates to a text file using the shared processing function
    to ensure 100% consistency with visualization coordinates.
    If USE_YOLO_FORMAT is True:
        Format: class_id x_center y_center width height (all normalized to 0-1)
        File naming: img_{image_num + 1}.txt
    Otherwise:
        Format: x1,y1,x2,y2,x3,y3,x4,y4,text
        File naming: gt_img_{image_num + 1}.txt
        where (x1,y1) is top-left, (x2,y2) is top-right,
              (x3,y3) is bottom-right, (x4,y4) is bottom-left
    All coordinates are rounded to integers for non-YOLO format
    """
    # Determine output directory based on whether it's a test or training image
    output_dir = TEST_LABELS_DIR if is_test else TRAIN_LABELS_DIR
    # Use different naming scheme based on format
    filename = f'img_{image_num + 1}.txt' if USE_YOLO_FORMAT else f'gt_img_{image_num + 1}.txt'
    output_file = os.path.join(output_dir, filename)
    
    # Process bounding boxes using the shared processing function
    processed_boxes = process_bounding_boxes_for_format(bounding_boxes)
    
    with open(output_file, 'w') as f:
        for processed_box in processed_boxes:
            if USE_YOLO_FORMAT:
                class_id, x_center, y_center, width, height = processed_box
                # Write in YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                x1, y1, x2, y2, x3, y3, x4, y4 = processed_box
                f.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},text\n")

def visualize_bounding_boxes(image, bounding_boxes, image_num, is_test=False):
    """
    Create a visualization of the bounding boxes using the EXACT SAME coordinates that are saved to .txt files.
    This function uses the shared processing function to ensure 100% consistency with saved coordinates.
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    vis_draw = ImageDraw.Draw(vis_image)
    
    # Process bounding boxes using the EXACT same function as save_bounding_boxes
    processed_boxes = process_bounding_boxes_for_format(bounding_boxes)
    
    # Draw each bounding box using the processed coordinates
    for processed_box in processed_boxes:
        if USE_YOLO_FORMAT:
            # Convert YOLO format back to pixel coordinates for visualization
            class_id, x_center, y_center, width, height = processed_box
            
            # Convert normalized coordinates back to pixel coordinates
            img_width = CANVAS_WIDTH
            img_height = CANVAS_HEIGHT
            
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            # Draw rectangle
            vis_draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=BOUNDING_BOX_COLOR,
                width=BOUNDING_BOX_LINE_WIDTH
            )
        else:
            # For polygon format, convert 8-point to rectangle for visualization
            x1, y1, x2, y2, x3, y3, x4, y4 = processed_box
            
            # Use the 4 points to draw the bounding rectangle
            # (x1,y1) = top-left, (x3,y3) = bottom-right
            vis_draw.rectangle(
                [(x1, y1), (x3, y3)],
                outline=BOUNDING_BOX_COLOR,
                width=BOUNDING_BOX_LINE_WIDTH
            )
    
    # Determine output directory based on whether it's a test or training image
    output_dir = TEST_IMAGES_DIR if is_test else TRAIN_IMAGES_DIR
    # Save the visualization with a different name to avoid overwriting original image
    vis_filename = f'vis_img_{image_num + 1}.jpg'
    vis_image.save(os.path.join(output_dir, vis_filename), quality=QUALITY)

def verify_coordinate_consistency(bounding_boxes, image_num, is_test=False):
    """
    Verify that the coordinates used for visualization match exactly with those saved to txt file.
    This is a debugging function to ensure 100% consistency.
    """
    # Process boxes for both save and visualization
    processed_boxes = process_bounding_boxes_for_format(bounding_boxes)
    
    # Print first few boxes for verification
    print(f"\nCoordinate Consistency Check for image {image_num + 1}:")
    for i, processed_box in enumerate(processed_boxes[:3]):  # Show first 3 boxes
        if USE_YOLO_FORMAT:
            class_id, x_center, y_center, width, height = processed_box
            print(f"  Box {i+1} (YOLO): class={class_id}, center=({x_center:.4f},{y_center:.4f}), size=({width:.4f},{height:.4f})")
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = processed_box
            print(f"  Box {i+1} (Polygon): ({x1},{y1})->({x2},{y2})->({x3},{y3})->({x4},{y4})")
            # Also show as rectangle for clarity
            width = x3 - x1
            height = y3 - y1
            print(f"    Rectangle: top-left=({x1},{y1}), size=({width}x{height})")

def main():
    # Calculate number of test and training images
    num_test_images = max(1, int(NUM_IMAGES * TEST_SPLIT_RATIO))  # Ensure at least 1 test image
    num_train_images = NUM_IMAGES - num_test_images
    
    # Generate specified number of images
    for image_num in range(NUM_IMAGES):
        # Determine if this is a test image
        is_test = image_num >= num_train_images
        
        # Create a new image with white background
        image = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        draw = ImageDraw.Draw(image)
        
        # Store all bounding boxes for this image
        all_bounding_boxes = []
        
        if USE_MULTIPLE_STRUCTURES == False:
            # Generate single structure
            all_bounding_boxes.extend(generate_structure(draw, 0, CANVAS_WIDTH, 0, CANVAS_HEIGHT, 0, 1, image))
        elif USE_MULTIPLE_STRUCTURES == True:
            # Always use multiple structures (2 or 3)
            num_structures = random.randint(2, 3)
            
            # Generate multiple structures
            for structure_idx in range(num_structures):
                # Calculate bounds for this structure
                left, right, top, bottom = calculate_structure_bounds(
                    CANVAS_WIDTH,
                    CANVAS_HEIGHT,
                    num_structures,
                    structure_idx
                )
                
                # Draw structure boundaries (invisible)
                draw.rectangle(
                    [(left, top), (right, bottom)],
                    outline='white',
                    width=1
                )
                
                # Generate structure with its index and total count
                all_bounding_boxes.extend(generate_structure(draw, left, right, top, bottom, structure_idx, num_structures, image))
        else:
            # USE_MULTIPLE_STRUCTURES is "RANDOM"
            # Randomly choose between 1, 2, or 3 structures
            num_structures = random.randint(1, 3)
            
            # Generate multiple structures
            for structure_idx in range(num_structures):
                # Calculate bounds for this structure
                left, right, top, bottom = calculate_structure_bounds(
                    CANVAS_WIDTH,
                    CANVAS_HEIGHT,
                    num_structures,
                    structure_idx
                )
                
                # Draw structure boundaries (invisible)
                draw.rectangle(
                    [(left, top), (right, bottom)],
                    outline='white',
                    width=1
                )
                
                # Generate structure with its index and total count
                all_bounding_boxes.extend(generate_structure(draw, left, right, top, bottom, structure_idx, num_structures, image))
        
        # Determine output directory based on whether it's a test or training image
        output_dir = TEST_IMAGES_DIR if is_test else TRAIN_IMAGES_DIR
        
        # Save the image as JPG with high quality and DPI settings
        image.save(os.path.join(output_dir, f'img_{image_num + 1}.jpg'), 
                  quality=QUALITY)
        
        # Save bounding boxes if enabled
        if SAVE_BOUNDING_BOXES:
            save_bounding_boxes(all_bounding_boxes, image_num, is_test)
        
        # Create bounding box visualization if enabled
        if VISUALIZE_BOUNDING_BOXES:
            visualize_bounding_boxes(image, all_bounding_boxes, image_num, is_test)
        
        # Verify coordinate consistency
        verify_coordinate_consistency(all_bounding_boxes, image_num, is_test)
        
        print(f"Generated image {image_num + 1}/{NUM_IMAGES} ({'test' if is_test else 'training'})")

if __name__ == "__main__":
    main()
