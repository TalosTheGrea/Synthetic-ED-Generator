import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Output configuration
OUTPUT_DIR = "generated_eds"
NUM_IMAGES = 1500 # Number of images to generate
START_NUMBER = 5000  # Starting number for file naming
TEST_SPLIT_RATIO = 0.01 # 10% of images will be used for testing
USE_YOLO_FORMAT = False  # Whether to save labels in YOLO format
YOLO_CLASS_ID = 0  # Class ID for text in YOLO format
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create directory structure
TRAIN_IMAGES_DIR = os.path.join(OUTPUT_DIR, "training_images")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_DIR, "training_labels_gt")
TEST_IMAGES_DIR = os.path.join(OUTPUT_DIR, "test_images")
TEST_LABELS_DIR = os.path.join(OUTPUT_DIR, "test_labels_gt")

# Create directories
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_LABELS_DIR, exist_ok=True)

# Performance optimization configuration
MAX_STRUCTURE_PLACEMENT_ATTEMPTS = 50  # Reduced from 100 to speed up structure placement
MAX_SYMBOL_POSITION_GRID = 10 # Reduced grid resolution for symbol positioning (was 10)
ENABLE_EARLY_TERMINATION = True  # Enable early termination when target area is reached
CACHE_LOADED_ASSETS = True  # Cache loaded structures and symbols
VERBOSE_OUTPUT = False  # Reduce console output for better performance
MAX_REMAINING_SPACES = 200  # Limit number of spaces to process for symbols

# Canvas configuration
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 640
QUALITY = 100

# Structure configuration
STRUCTURES_FOLDER = "elements/structures"
SYMBOLS_FOLDER = "elements/symbols"
STRUCTURE_RESIZE_RATIO = 0.4  # 40% of canvas size
SYMBOL_RESIZE_RATIO = 0.1    # 15% of canvas size
COMPACTNESS = 0.7   # 50% of canvas to be filled
# Intersection thresholds
STRUCTURE_INTERSECT_THRESHOLD = 0.3  # How much structures can intersect with each other (30% allowed)
SYMBOL_STRUCTURE_INTERSECT_THRESHOLD = 0.0  # How much symbols+text can intersect with structures (10% allowed)

# Spacing configuration
MIN_SYMBOL_SPACING = 3  # Minimum distance between symbol-text units
SYMBOL_TEXT_PADDING = 3  # Padding between symbol and its text
SYMBOL_EDGE_PADDING = 10  # Minimum distance from canvs edges

# Bounding box configuration
BOUNDING_BOX_PADDING = 1.5  # Padding around text for bounding box
SAVE_BOUNDING_BOXES = True  # Whether to save bounding box coordinates
VISUALIZE_BOUNDING_BOXES = False  # Whether to create visualization of bounding boxes
BOUNDING_BOX_COLOR = 'red'  # Color for visualizing bounding boxes
BOUNDING_BOX_LINE_WIDTH = 1  # Line width for visualizing bounding boxes

# Text configuration
MIN_TEXT_LENGTH = 4    # Minimum length of text
MAX_TEXT_LENGTH = 6    # Maximum length of text
NUMBER_RATIO = 0.6     # Ratio of numbers in the text
FONT_SIZE = 20       # Base font size
FONT_DIR = "fonts"     # Directory containing fonts
TEXT_SPACING = 5      # Spacing between text and symbol
USE_SPECIAL_SYMBOLS = True  # Whether to use special symbols in text
SPECIAL_SYMBOL_PROBABILITY = 1  # Probability of adding a special symbol to text

# Multi-row text configuration
MIN_TEXT_ROWS = 1      # Minimum number of rows for text
MAX_TEXT_ROWS = 2      # Maximum number of rows for text
TEXT_ROW_SPACING = 1   # Vertical spacing between text rows (in pixels)

# Text splitting configuration
TEXT_SPLIT_PROBABILITY = 0.3  # Probability that a text row will be split into 2 segments
TEXT_SEGMENT_SPACING = 8      # Horizontal spacing between split text segments (in pixels)

# Simple text overlap prevention configuration
PREVENT_TEXT_OVERLAPS = True  # Enable simple text overlap prevention
TEXT_OVERLAP_TOLERANCE = 0.05  # Allow up to 10% overlap between text segments (0.0 = no overlap, 1.0 = full overlap allowed)
TEXT_OVERLAP_PADDING = 2      # Extra padding around text for overlap detection

# White box background configuration
USE_WHITE_BOX_BACKGROUND = True  # Enable white box backgrounds behind text
WHITE_BOX_PADDING_X = 1          # Horizontal padding around text in white box (pixels)
WHITE_BOX_PADDING_Y = 0          # Vertical padding around text in white box (pixels)
WHITE_BOX_FILL_COLOR = 'white'   # Fill color for text background boxes
WHITE_BOX_OUTLINE_COLOR = None   # Outline color for text background boxes (None = no outline)
WHITE_BOX_OUTLINE_WIDTH = 0      # Outline width for text background boxes (0 = no outline)

# Rotation configuration
USE_RANDOM_ROTATION = True  # Whether to randomly rotate structures and symbols
ROTATION_PROBABILITY = 0.5  # Probability that an element will be rotated (50%)
AVAILABLE_ROTATIONS = [90, 180, 270]  # Available rotation angles in degrees

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

# Create necessary folders if they don't exist
os.makedirs(STRUCTURES_FOLDER, exist_ok=True)
os.makedirs(SYMBOLS_FOLDER, exist_ok=True)
os.makedirs(FONT_DIR, exist_ok=True)

# Global cache for loaded assets
_cached_structures = None
_cached_symbols = None
_cached_fonts = None

# Global tracking for text overlap prevention
_placed_text_boxes = []
_skipped_text_segments = 0

def reset_text_tracking():
    """Reset the global text tracking for a new image generation"""
    global _placed_text_boxes, _skipped_text_segments
    _placed_text_boxes = []
    _skipped_text_segments = 0

def get_text_overlap_stats():
    """Get statistics about text overlap prevention"""
    return {
        'placed_text_boxes': len(_placed_text_boxes),
        'skipped_text_segments': _skipped_text_segments
    }

def calculate_simple_overlap(box1, box2):
    """
    Calculate overlap percentage between two bounding boxes.
    Returns overlap as a percentage (0.0 to 1.0) relative to the smaller box.
    box format: (x1, y1, x2, y2)
    """
    # Calculate intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate overlap percentage relative to the smaller box
    smaller_area = min(box1_area, box2_area)
    if smaller_area <= 0:
        return 0.0
    
    return intersection_area / smaller_area

def check_text_overlap(new_text_box):
    """
    Check if a new text box would overlap with existing text beyond tolerance.
    Returns True if placement is valid (no significant overlap), False otherwise.
    """
    if not PREVENT_TEXT_OVERLAPS:
        return True
    
    # Add padding to the new text box for overlap detection
    x1, y1, x2, y2 = new_text_box
    padded_box = (x1 - TEXT_OVERLAP_PADDING, y1 - TEXT_OVERLAP_PADDING, 
                  x2 + TEXT_OVERLAP_PADDING, y2 + TEXT_OVERLAP_PADDING)
    
    # Check against all existing text boxes
    for existing_box in _placed_text_boxes:
        overlap = calculate_simple_overlap(padded_box, existing_box)
        if overlap > TEXT_OVERLAP_TOLERANCE:
            return False
    
    return True

def add_text_box(text_box):
    """Add a text box to the global tracking list"""
    if PREVENT_TEXT_OVERLAPS:
        _placed_text_boxes.append(text_box)

def load_structures(structures_folder=STRUCTURES_FOLDER):
    """Load all structure images from the specified folder with caching"""
    global _cached_structures
    
    if CACHE_LOADED_ASSETS and _cached_structures is not None:
        return _cached_structures
    
    structures_path = Path(structures_folder)
    if not structures_path.exists():
        raise FileNotFoundError(f"Structures folder not found: {structures_folder}")
        
    structures = []
    for img_file in structures_path.glob('*'):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                img = Image.open(img_file)
                structures.append(img)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print(f"Loaded {len(structures)} structures from {structures_folder}")
    
    if CACHE_LOADED_ASSETS:
        _cached_structures = structures
    
    return structures

def load_symbols(symbols_folder=SYMBOLS_FOLDER):
    """Load all symbol images from the specified folder with caching"""
    global _cached_symbols
    
    if CACHE_LOADED_ASSETS and _cached_symbols is not None:
        return _cached_symbols
    
    symbols_path = Path(symbols_folder)
    if not symbols_path.exists():
        raise FileNotFoundError(f"Symbols folder not found: {symbols_folder}")
        
    symbols = []
    for img_file in symbols_path.glob('*'):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                img = Image.open(img_file)
                symbols.append(img)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print(f"Loaded {len(symbols)} symbols from {symbols_folder}")
    
    if CACHE_LOADED_ASSETS:
        _cached_symbols = symbols
    
    return symbols

def resize_structure(img, canvas_width, canvas_height, resize_ratio):
    """Resize structure image while maintaining aspect ratio"""
    target_width = int(canvas_width * resize_ratio)
    target_height = int(canvas_height * resize_ratio)
    
    # Calculate aspect ratio
    aspect_ratio = img.width / img.height
    
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def apply_random_rotation(img):
    """Apply random rotation to an image if rotation is enabled"""
    if not USE_RANDOM_ROTATION:
        return img
    
    if random.random() < ROTATION_PROBABILITY:
        rotation_angle = random.choice(AVAILABLE_ROTATIONS)
        # Rotate the image and expand to fit new dimensions
        rotated_img = img.rotate(rotation_angle, expand=True)
        return rotated_img
    
    return img

def calculate_intersection(rect1, rect2):
    """Calculate intersection area between two rectangles"""
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    rect1_area = rect1[2] * rect1[3]
    rect2_area = rect2[2] * rect2[3]
    
    # Return the maximum intersection ratio
    return max(intersection_area / rect1_area, intersection_area / rect2_area)

def is_valid_placement(new_rect, placed_rects, intersect_threshold):
    """Check if the new rectangle placement is valid"""
    for rect in placed_rects:
        if calculate_intersection(new_rect, rect) > intersect_threshold:
            return False
    return True

def generate_single_text_line():
    """Generate a single line of text with specified number and letter ratio"""
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
    """Generate multi-row text with specified number and letter ratio, potentially splitting some rows"""
    # Generate multiple rows of text
    num_rows = random.randint(MIN_TEXT_ROWS, MAX_TEXT_ROWS)
    text_segments = []  # Each element is (text, row_index, segment_index)
    
    for row_idx in range(num_rows):
        base_text = generate_single_text_line()
        
        # Decide if this row should be split into 2 segments
        if random.random() < TEXT_SPLIT_PROBABILITY and len(base_text) >= 4:  # Only split if text is long enough
            # Split the text roughly in the middle
            split_point = len(base_text) // 2 + random.randint(-1, 1)  # Add some randomness
            split_point = max(1, min(len(base_text) - 1, split_point))  # Ensure valid split
            
            segment1 = base_text[:split_point]
            segment2 = base_text[split_point:]
            
            text_segments.append((segment1, row_idx, 0))  # First segment of the row
            text_segments.append((segment2, row_idx, 1))  # Second segment of the row
        else:
            # Keep as single segment
            text_segments.append((base_text, row_idx, 0))
    
    return text_segments

def get_available_text_positions(symbol_rect, placed_rects, canvas_width, canvas_height):
    """Get available positions for text around the symbol"""
    available_positions = []
    
    # Calculate maximum text dimensions for multi-row text (accounting for potential splitting)
    max_text_width = FONT_SIZE * 10  # Increased to accommodate split text with spacing
    max_text_height = MAX_TEXT_ROWS * FONT_SIZE + (MAX_TEXT_ROWS - 1) * TEXT_ROW_SPACING
    
    # Define potential text positions (left, right, top, bottom)
    positions = [
        ('left', (symbol_rect[0] - max_text_width, symbol_rect[1], max_text_width, max_text_height)),
        ('right', (symbol_rect[0] + symbol_rect[2], symbol_rect[1], max_text_width, max_text_height)),
        ('top', (symbol_rect[0], symbol_rect[1] - max_text_height, symbol_rect[2], max_text_height)),
        ('bottom', (symbol_rect[0], symbol_rect[1] + symbol_rect[3], symbol_rect[2], max_text_height))
    ]
    
    for pos_name, text_rect in positions:
        # Check if text would be within canvas bounds
        if (text_rect[0] >= 0 and text_rect[0] + text_rect[2] <= canvas_width and
            text_rect[1] >= 0 and text_rect[1] + text_rect[3] <= canvas_height):
            # Check if text would overlap with any placed elements
            if is_valid_placement(text_rect, placed_rects, 0):
                available_positions.append(pos_name)
    
    return available_positions

def place_text(draw, text_segments, position, symbol_rect, font):
    """Place text segments at the specified position relative to the symbol, returning individual bounding boxes"""
    
    # Group segments by row
    rows = {}
    for text, row_idx, segment_idx in text_segments:
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append((text, segment_idx))
    
    # Sort rows by index
    sorted_rows = sorted(rows.items())
    
    # Calculate total dimensions for positioning
    max_row_width = 0
    for row_idx, segments in sorted_rows:
        if len(segments) == 1:
            # Single segment
            row_width = font.getlength(segments[0][0])
        else:
            # Multiple segments - add spacing between them
            row_width = sum(font.getlength(seg[0]) for seg, _ in segments) + TEXT_SEGMENT_SPACING * (len(segments) - 1)
        max_row_width = max(max_row_width, row_width)
    
    total_text_height = len(sorted_rows) * FONT_SIZE + (len(sorted_rows) - 1) * TEXT_ROW_SPACING
    
    # Calculate starting position based on text position relative to symbol
    if position == 'left':
        base_x = symbol_rect[0] - max_row_width - SYMBOL_TEXT_PADDING
        base_y = symbol_rect[1] + (symbol_rect[3] - total_text_height) // 2
    elif position == 'right':
        base_x = symbol_rect[0] + symbol_rect[2] + SYMBOL_TEXT_PADDING
        base_y = symbol_rect[1] + (symbol_rect[3] - total_text_height) // 2
    elif position == 'top':
        base_x = symbol_rect[0] + (symbol_rect[2] - max_row_width) // 2
        base_y = symbol_rect[1] - total_text_height - SYMBOL_TEXT_PADDING
    else:  # bottom
        base_x = symbol_rect[0] + (symbol_rect[2] - max_row_width) // 2
        base_y = symbol_rect[1] + symbol_rect[3] + SYMBOL_TEXT_PADDING
    
    # Draw each segment and collect individual bounding boxes
    individual_bounding_boxes = []
    current_y = base_y
    
    for row_idx, segments in sorted_rows:
        # Sort segments by segment index within the row
        segments.sort(key=lambda x: x[1])
        
        # Calculate row width for centering
        if len(segments) == 1:
            row_width = font.getlength(segments[0][0])
        else:
            row_width = sum(font.getlength(seg[0]) for seg, _ in segments) + TEXT_SEGMENT_SPACING * (len(segments) - 1)
        
        # Calculate starting x for this row (centered if top/bottom position)
        if position in ['top', 'bottom']:
            row_start_x = base_x + (max_row_width - row_width) // 2
        else:
            row_start_x = base_x
        
        # Place each segment in the row
        current_x = row_start_x
        for text, segment_idx in segments:
            segment_width = font.getlength(text)
            
            # Create bounding box for this individual segment
            segment_bbox = (current_x, current_y, current_x + segment_width, current_y + FONT_SIZE)
            
            # Check for text overlap before placing
            if check_text_overlap(segment_bbox):
                # Draw white box background if enabled
                if USE_WHITE_BOX_BACKGROUND:
                    # Calculate white box dimensions with padding
                    white_box_x1 = current_x - WHITE_BOX_PADDING_X
                    white_box_y1 = current_y - WHITE_BOX_PADDING_Y
                    white_box_x2 = current_x + segment_width + WHITE_BOX_PADDING_X
                    white_box_y2 = current_y + FONT_SIZE + WHITE_BOX_PADDING_Y
                    
                    # Draw the white box background
                    draw.rectangle(
                        [(white_box_x1, white_box_y1), (white_box_x2, white_box_y2)],
                        fill=WHITE_BOX_FILL_COLOR,
                        outline=WHITE_BOX_OUTLINE_COLOR,
                        width=WHITE_BOX_OUTLINE_WIDTH
                    )
                
                # Draw the text segment on top of the white box
                draw.text((current_x, current_y), text, fill='black', font=font)
                individual_bounding_boxes.append(segment_bbox)
                
                # Add to global tracking
                add_text_box(segment_bbox)
            else:
                # Skip this segment due to overlap
                global _skipped_text_segments
                _skipped_text_segments += 1
                if VERBOSE_OUTPUT:
                    print(f"Skipping text segment '{text}' due to overlap")
            
            # Move to next segment position regardless of whether text was placed
            current_x += segment_width + TEXT_SEGMENT_SPACING
        
        # Move to next row
        current_y += FONT_SIZE + TEXT_ROW_SPACING
    
    # Return individual bounding boxes and overall dimensions for compatibility
    overall_bounds = (base_x, base_y, max_row_width, total_text_height)
    return individual_bounding_boxes, overall_bounds

def get_remaining_spaces(placed_rects, canvas_width, canvas_height, min_size=50):
    """Get available spaces for placing symbols based on placed rectangles"""
    # Start with the entire canvas
    spaces = [(0, 0, canvas_width, canvas_height)]
    
    # For each placed rectangle, split the spaces
    for rect in placed_rects:
        new_spaces = []
        for space in spaces:
            # Check if space overlaps with rectangle
            if (rect[0] < space[0] + space[2] and rect[0] + rect[2] > space[0] and
                rect[1] < space[1] + space[3] and rect[1] + rect[3] > space[1]):
                
                # Split space into smaller spaces
                # Left space
                if rect[0] > space[0]:
                    new_spaces.append((space[0], space[1], rect[0] - space[0], space[3]))
                
                # Right space
                if rect[0] + rect[2] < space[0] + space[2]:
                    new_spaces.append((rect[0] + rect[2], space[1], 
                                     space[0] + space[2] - (rect[0] + rect[2]), space[3]))
                
                # Top space
                if rect[1] > space[1]:
                    new_spaces.append((space[0], space[1], space[2], rect[1] - space[1]))
                
                # Bottom space
                if rect[1] + rect[3] < space[1] + space[3]:
                    new_spaces.append((space[0], rect[1] + rect[3], 
                                     space[2], space[1] + space[3] - (rect[1] + rect[3])))
            else:
                new_spaces.append(space)
        
        spaces = new_spaces
    
    # Filter out spaces that are too small
    return [space for space in spaces if space[2] >= min_size and space[3] >= min_size]

def is_valid_symbol_placement(new_rect, placed_rects, canvas_width, canvas_height):
    """Check if the new symbol placement is valid (no intersections with any elements)"""
    # Check if symbol is within canvas bounds
    if (new_rect[0] < 0 or new_rect[1] < 0 or 
        new_rect[0] + new_rect[2] > canvas_width or 
        new_rect[1] + new_rect[3] > canvas_height):
        return False
    
    # Check for any intersection with placed rectangles
    for rect in placed_rects:
        if calculate_intersection(new_rect, rect) > 0:
            return False
    
    return True

def is_valid_symbol_text_vs_structures(symbol_rect, text_rect, position, structure_rects, canvas_width, canvas_height, intersect_threshold):
    """
    Check if the complete symbol+text unit is valid against structures.
    This checks the ENTIRE combined bounds of symbol+text against placed structures.
    """
    # Calculate the combined bounds of symbol+text
    combined_bounds = calculate_symbol_text_bounds(symbol_rect, text_rect, position)
    
    # Check if combined bounds are within canvas
    if (combined_bounds[0] < 0 or combined_bounds[1] < 0 or 
        combined_bounds[2] > canvas_width or combined_bounds[3] > canvas_height):
        return False
    
    # Convert combined bounds to rectangle format (x, y, width, height)
    combined_rect = (combined_bounds[0], combined_bounds[1], 
                    combined_bounds[2] - combined_bounds[0], 
                    combined_bounds[3] - combined_bounds[1])
    
    # Check intersection with all structure rectangles
    for structure_rect in structure_rects:
        intersection_ratio = calculate_intersection(combined_rect, structure_rect)
        if intersection_ratio > intersect_threshold:
            return False
    
    return True

def get_random_font(font_size=FONT_SIZE):
    """Get a random font from the fonts directory with caching"""
    global _cached_fonts
    
    try:
        # Initialize font cache if needed
        if CACHE_LOADED_ASSETS and _cached_fonts is None:
            font_files = list(Path(FONT_DIR).glob('*.ttf')) + list(Path(FONT_DIR).glob('*.otf'))
            if not font_files:
                print("No font files found in fonts directory, using default font")
                _cached_fonts = []
            else:
                _cached_fonts = font_files
        
        # Use cached font files or load fresh
        font_files = _cached_fonts if CACHE_LOADED_ASSETS else (list(Path(FONT_DIR).glob('*.ttf')) + list(Path(FONT_DIR).glob('*.otf')))
        
        if not font_files:
            return ImageFont.load_default()
        
        # Select a random font file
        font_path = random.choice(font_files)
        return ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        print(f"Error loading font: {e}, using default font")
        return ImageFont.load_default()

def get_valid_symbol_positions(space, symbol_size, placed_rects, canvas_width, canvas_height):
    """Get all valid positions for a symbol within a given space"""
    valid_positions = []
    
    # Calculate grid of possible positions with optimized grid size
    step_x = max(1, space[2] // MAX_SYMBOL_POSITION_GRID)
    step_y = max(1, space[3] // MAX_SYMBOL_POSITION_GRID)
    
    for x in range(space[0], space[0] + space[2] - symbol_size[0], step_x):
        for y in range(space[1], space[1] + space[3] - symbol_size[1], step_y):
            new_rect = (x, y, symbol_size[0], symbol_size[1])
            if is_valid_symbol_placement(new_rect, placed_rects, canvas_width, canvas_height):
                valid_positions.append(new_rect)
    
    return valid_positions

def calculate_visible_area(rect, canvas_width, canvas_height):
    """Calculate the visible area of a rectangle within the canvas bounds"""
    x1 = max(0, rect[0])
    y1 = max(0, rect[1])
    x2 = min(canvas_width, rect[0] + rect[2])
    y2 = min(canvas_height, rect[1] + rect[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    return (x2 - x1) * (y2 - y1)

def add_padding_to_box(box, padding):
    """
    Add padding to a bounding box
    box: (x1, y1, x2, y2)
    padding: amount of padding to add
    Returns: (x1, y1, x2, y2) with padding
    """
    x1, y1, x2, y2 = box
    return (x1 - padding, y1 - padding, x2 + padding, y2 + padding)

def validate_bounding_boxes(bounding_boxes, canvas_width, canvas_height):
    """
    Validate that all bounding boxes have correct format and are within bounds
    """
    valid_boxes = []
    for i, box in enumerate(bounding_boxes):
        try:
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are numbers
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Check if box is within canvas bounds (allowing for some tolerance)
            if x1 >= -1 and y1 >= -1 and x2 <= canvas_width + 1 and y2 <= canvas_height + 1:
                # Ensure box has positive area
                if x2 > x1 and y2 > y1:
                    valid_boxes.append((x1, y1, x2, y2))
                else:
                    print(f"Warning: Box {i} has zero or negative area: {box}")
            else:
                print(f"Warning: Box {i} is outside canvas bounds: {box}")
                
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid box format at index {i}: {box}, error: {e}")
    
    return valid_boxes

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

def save_bounding_boxes(bounding_boxes, image_num, is_test=False):
    """
    Save bounding box coordinates to a text file
    If USE_YOLO_FORMAT is True:
        Format: class_id x_center y_center width height (all normalized to 0-1)
        File naming: img_{START_NUMBER + image_num}.txt
    Otherwise:
        Format: x1,y1,x2,y2,x3,y3,x4,y4,text
        File naming: gt_img_{START_NUMBER + image_num}.txt
        where (x1,y1) is top-left, (x2,y2) is top-right,
              (x3,y3) is bottom-right, (x4,y4) is bottom-left
    All coordinates are rounded to integers for non-YOLO format
    """
    # Determine output directory based on whether it's a test or training image
    output_dir = TEST_LABELS_DIR if is_test else TRAIN_LABELS_DIR
    # Use different naming scheme based on format and include starting number
    filename = f'img_{START_NUMBER + image_num}.txt' if USE_YOLO_FORMAT else f'gt_img_{START_NUMBER + image_num}.txt'
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
    vis_filename = f'vis_img_{START_NUMBER + image_num}.jpg'
    vis_image.save(os.path.join(output_dir, vis_filename), quality=QUALITY)

def calculate_symbol_text_bounds(symbol_rect, text_rect, position):
    """
    Calculate the combined bounds of a symbol and its text
    Returns: (x1, y1, x2, y2) representing the total area occupied
    """
    symbol_x1, symbol_y1 = symbol_rect[0], symbol_rect[1]
    symbol_x2 = symbol_x1 + symbol_rect[2]
    symbol_y2 = symbol_y1 + symbol_rect[3]
    
    text_x1, text_y1 = text_rect[0], text_rect[1]
    text_x2 = text_x1 + text_rect[2]
    text_y2 = text_y1 + text_rect[3]
    
    if position == 'left':
        return (text_x1, min(symbol_y1, text_y1), 
                symbol_x2, max(symbol_y2, text_y2))
    elif position == 'right':
        return (symbol_x1, min(symbol_y1, text_y1), 
                text_x2, max(symbol_y2, text_y2))
    elif position == 'top':
        return (min(symbol_x1, text_x1), text_y1, 
                max(symbol_x2, text_x2), symbol_y2)
    else:  # bottom
        return (min(symbol_x1, text_x1), symbol_y1, 
                max(symbol_x2, text_x2), text_y2)

def is_valid_symbol_text_placement(symbol_rect, text_rect, position, placed_units, canvas_width, canvas_height):
    """
    Check if the symbol-text unit can be placed without violating spacing rules
    """
    # Calculate combined bounds
    bounds = calculate_symbol_text_bounds(symbol_rect, text_rect, position)
    
    # Check canvas edge spacing
    if (bounds[0] < SYMBOL_EDGE_PADDING or 
        bounds[1] < SYMBOL_EDGE_PADDING or 
        bounds[2] > canvas_width - SYMBOL_EDGE_PADDING or 
        bounds[3] > canvas_height - SYMBOL_EDGE_PADDING):
        return False
    
    # Check spacing with other placed units
    for unit_bounds in placed_units:
        # Calculate minimum distance between units
        x_dist = max(0, unit_bounds[0] - bounds[2], bounds[0] - unit_bounds[2])
        y_dist = max(0, unit_bounds[1] - bounds[3], bounds[1] - unit_bounds[3])
        
        # If either distance is less than minimum spacing, placement is invalid
        if x_dist < MIN_SYMBOL_SPACING and y_dist < MIN_SYMBOL_SPACING:
            return False
    
    return True

def generate_drawing(canvas_width=CANVAS_WIDTH, canvas_height=CANVAS_HEIGHT, quality=QUALITY,
                    structure_resize_ratio=STRUCTURE_RESIZE_RATIO, 
                    symbol_resize_ratio=SYMBOL_RESIZE_RATIO,
                    compactness=COMPACTNESS, 
                    structure_intersect_threshold=STRUCTURE_INTERSECT_THRESHOLD,
                    symbol_structure_intersect_threshold=SYMBOL_STRUCTURE_INTERSECT_THRESHOLD):
    """Generate the engineering drawing with placed structures and symbols"""
    # Reset text tracking for this new image
    reset_text_tracking()
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Initialize list to store all bounding boxes
    all_bounding_boxes = []
    
    # Load structures and symbols
    structures = load_structures()
    symbols = load_symbols()
    if not structures:
        raise ValueError("No structure images loaded")
    if not symbols:
        raise ValueError("No symbol images loaded")
    
    # Initialize tracking variables
    placed_structures = []  # Track structure rectangles specifically
    placed_rects = []       # Track all rectangles (structures + symbols)
    placed_units = []       # Track symbol-text units for spacing
    total_area = canvas_width * canvas_height
    target_area = total_area * compactness
    current_area = 0
    
    # Create a list of all possible structure indices
    # This ensures we use all structures before repeating
    available_structures = list(range(len(structures)))
    random.shuffle(available_structures)  # Shuffle once at the start
    
    # First, place structures
    while current_area < target_area:
        # If we've used all structures, reshuffle the list
        if not available_structures:
            available_structures = list(range(len(structures)))
            random.shuffle(available_structures)
        
        # Get next structure from our shuffled list
        structure_idx = available_structures.pop()
        structure = structures[structure_idx]
        
        resized = resize_structure(structure, canvas_width, canvas_height, structure_resize_ratio)
        resized = apply_random_rotation(resized)
        
        # Try to place the structure
        placed = False
        for _ in range(MAX_STRUCTURE_PLACEMENT_ATTEMPTS):  # Optimized placement attempts
            # Random position (can be outside canvas)
            x = random.randint(-resized.width, canvas_width)
            y = random.randint(-resized.height, canvas_height)
            
            new_rect = (x, y, resized.width, resized.height)
            
            # Calculate visible area of this placement
            visible_area = calculate_visible_area(new_rect, canvas_width, canvas_height)
            
            # Check if this placement would help us reach the target area
            if current_area + visible_area > target_area * 1.1:  # Allow 10% overflow
                continue
            
            if is_valid_placement(new_rect, placed_structures, structure_intersect_threshold):
                # Paste the structure
                canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)
                placed_structures.append(new_rect)  # Track as structure
                placed_rects.append(new_rect)       # Track in all rectangles
                current_area += visible_area
                placed = True
                if VERBOSE_OUTPUT and len(placed_rects) % 5 == 0:  # Print every 5th structure to reduce output
                    print(f"Placed structure {structure_idx + 1}, current coverage: {current_area/total_area:.1%}")
                break
        
        if not placed:
            print(f"Could not place structure {structure_idx + 1}, moving to symbol placement")
            print(f"Final structure coverage: {current_area/total_area:.1%}")
            break
    
    print(f"Placed {len(placed_structures)} structures, covering {current_area/total_area:.1%} of canvas")
    
    # Get remaining spaces for symbols (based on structures only, since symbols can overlap other symbols)
    remaining_spaces = get_remaining_spaces(placed_structures, canvas_width, canvas_height)
    
    # Limit number of spaces to process for performance
    if len(remaining_spaces) > MAX_REMAINING_SPACES:
        # Sort by area (largest first) and take the best spaces
        remaining_spaces.sort(key=lambda space: space[2] * space[3], reverse=True)
        remaining_spaces = remaining_spaces[:MAX_REMAINING_SPACES]
    
    if VERBOSE_OUTPUT:
        print(f"Found {len(remaining_spaces)} available spaces for symbols")
    
    # Calculate remaining area to fill based on actual achieved density
    achieved_density = current_area / total_area
    remaining_area = total_area * (1 - achieved_density)
    if VERBOSE_OUTPUT:
        print(f"Using achieved density of {achieved_density:.1%} for symbol placement")
        print(f"Remaining area for symbols: {remaining_area/total_area:.1%}")
    
    current_symbol_area = 0
    symbols_placed = 0
    
    # Store text placements for later drawing
    text_placements = []
    
    # Place symbols in remaining spaces
    max_symbols_per_space = 3  # Limit symbols per space for performance
    for space in remaining_spaces:
        if current_symbol_area >= remaining_area:
            break
        
        symbols_in_space = 0
            
        # Get all valid positions for symbols in this space
        # We'll get positions for an average symbol size first
        avg_symbol = random.choice(symbols)
        resized = resize_structure(avg_symbol, canvas_width, canvas_height, symbol_resize_ratio)
        resized = apply_random_rotation(resized)
        symbol_size = (resized.width, resized.height)
        
        valid_positions = get_valid_symbol_positions(space, symbol_size, placed_structures, canvas_width, canvas_height)
        
        # Try each valid position with a random symbol
        for pos in valid_positions:
            if current_symbol_area >= remaining_area or symbols_in_space >= max_symbols_per_space:
                break
                
            # Select a random symbol for this position
            symbol = random.choice(symbols)
            resized = resize_structure(symbol, canvas_width, canvas_height, symbol_resize_ratio)
            resized = apply_random_rotation(resized)
            
            # Adjust position if needed for the new symbol size
            if resized.width != symbol_size[0] or resized.height != symbol_size[1]:
                # Recalculate position to keep symbol centered in the original space
                x = pos[0] + (symbol_size[0] - resized.width) // 2
                y = pos[1] + (symbol_size[1] - resized.height) // 2
                pos = (x, y, resized.width, resized.height)
                
                # Check if the new position is still valid
                if not is_valid_symbol_placement(pos, placed_structures, canvas_width, canvas_height):
                    continue
            
            # Check available positions for text
            available_positions = get_available_text_positions(pos, placed_structures, canvas_width, canvas_height)
            
            if available_positions:
                # Generate text and calculate its position
                text_segments = generate_text()
                position = random.choice(available_positions)
                font = get_random_font()
                
                # Calculate text dimensions for segments (for validation)
                # Group segments by row to calculate dimensions
                rows = {}
                for text, row_idx, segment_idx in text_segments:
                    if row_idx not in rows:
                        rows[row_idx] = []
                    rows[row_idx].append((text, segment_idx))
                
                max_text_width = 0
                for row_idx, segments in rows.items():
                    if len(segments) == 1:
                        row_width = font.getlength(segments[0][0])
                    else:
                        row_width = sum(font.getlength(seg[0]) for seg, _ in segments) + TEXT_SEGMENT_SPACING * (len(segments) - 1)
                    max_text_width = max(max_text_width, row_width)
                
                total_text_height = len(rows) * FONT_SIZE + (len(rows) - 1) * TEXT_ROW_SPACING
                
                # Calculate text position without drawing
                if position == 'left':
                    x = pos[0] - max_text_width - SYMBOL_TEXT_PADDING
                    y = pos[1] + (pos[3] - total_text_height) // 2
                elif position == 'right':
                    x = pos[0] + pos[2] + SYMBOL_TEXT_PADDING
                    y = pos[1] + (pos[3] - total_text_height) // 2
                elif position == 'top':
                    x = pos[0] + (pos[2] - max_text_width) // 2
                    y = pos[1] - total_text_height - SYMBOL_TEXT_PADDING
                else:  # bottom
                    x = pos[0] + (pos[2] - max_text_width) // 2
                    y = pos[1] + pos[3] + SYMBOL_TEXT_PADDING
                
                text_rect = (x, y, max_text_width, total_text_height)
                
                # Check if the symbol-text unit can be placed with proper spacing AND doesn't intersect too much with structures
                spacing_valid = is_valid_symbol_text_placement(pos, text_rect, position, placed_units, canvas_width, canvas_height)
                structure_valid = is_valid_symbol_text_vs_structures(pos, text_rect, position, placed_structures, canvas_width, canvas_height, symbol_structure_intersect_threshold)
                
                if spacing_valid and structure_valid:
                    # Place symbol
                    canvas.paste(resized, (pos[0], pos[1]), resized if resized.mode == 'RGBA' else None)
                    placed_rects.append(pos)
                    
                    # Calculate and store the combined bounds
                    unit_bounds = calculate_symbol_text_bounds(pos, text_rect, position)
                    placed_units.append(unit_bounds)
                    
                    # Store text placement info for later drawing (bounding boxes will be created after drawing)
                    text_placements.append({
                        'text_segments': text_segments,
                        'position': position,
                        'symbol_rect': pos,
                        'font': font
                    })
                    
                    # Update area and count
                    symbol_area = pos[2] * pos[3]
                    text_area = text_rect[2] * text_rect[3]
                    current_symbol_area += symbol_area + text_area
                    symbols_placed += 1
                    symbols_in_space += 1
                    
                    if VERBOSE_OUTPUT and symbols_placed % 10 == 0:  # Print every 10th symbol to reduce output
                        print(f"Placed symbol {symbols_placed}, covering {current_symbol_area/remaining_area:.1%} of remaining area")
    
    # Draw all text at the end and collect individual bounding boxes
    total_text_segments = 0
    for placement in text_placements:
        individual_bboxes, overall_bounds = place_text(draw, placement['text_segments'], placement['position'], 
                                                      placement['symbol_rect'], placement['font'])
        
        # Debug: Print info about text segments for first placement
        if len(text_placements) > 0 and placement == text_placements[0]:
            print(f"Debug - First text placement has {len(placement['text_segments'])} segments:")
            for text, row_idx, segment_idx in placement['text_segments']:
                print(f"  Row {row_idx}, Segment {segment_idx}: '{text}'")
        
        # Add each individual segment bounding box with padding
        for bbox in individual_bboxes:
            bbox_with_padding = add_padding_to_box(bbox, BOUNDING_BOX_PADDING)
            
            # Ensure bounding box is within canvas bounds
            x1, y1, x2, y2 = bbox_with_padding
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(canvas_width, x2)
            y2 = min(canvas_height, y2)
            
            # Only add if the bounding box has positive area
            if x2 > x1 and y2 > y1:
                all_bounding_boxes.append((x1, y1, x2, y2))
                total_text_segments += 1
    
    print(f"Generated {total_text_segments} individual text segment bounding boxes")
    
    # Report text overlap prevention statistics
    if PREVENT_TEXT_OVERLAPS:
        stats = get_text_overlap_stats()
        total_attempted = stats['placed_text_boxes'] + stats['skipped_text_segments']
        if total_attempted > 0:
            skip_rate = (stats['skipped_text_segments'] / total_attempted) * 100
            print(f"Text overlap prevention: {stats['skipped_text_segments']} segments skipped ({skip_rate:.1f}% skip rate)")
    
    print(f"Placed {symbols_placed} symbols with text, covering {current_symbol_area/remaining_area:.1%} of remaining area")
    print(f"Final density: {(current_area + current_symbol_area)/total_area:.1%}")
    return canvas, all_bounding_boxes

def save_drawing(canvas, output_path, quality=QUALITY):
    """Save the generated drawing"""
    canvas.save(output_path, quality=quality)

def verify_coordinate_consistency(bounding_boxes, image_num, is_test=False):
    """
    Verify that the coordinates used for visualization match exactly with those saved to txt file.
    This is a debugging function to ensure 100% consistency.
    """
    # Process boxes for both save and visualization
    processed_boxes = process_bounding_boxes_for_format(bounding_boxes)
    
    # Print first few boxes for verification
    print(f"\nCoordinate Consistency Check for image {START_NUMBER + image_num}:")
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
        
        # Generate the drawing
        canvas, bounding_boxes = generate_drawing()
        
        # Validate bounding boxes
        validated_boxes = validate_bounding_boxes(bounding_boxes, CANVAS_WIDTH, CANVAS_HEIGHT)
        
        # Save the result with starting number
        output_dir = TEST_IMAGES_DIR if is_test else TRAIN_IMAGES_DIR
        output_path = os.path.join(output_dir, f'img_{START_NUMBER + image_num}.jpg')
        save_drawing(canvas, output_path)
        
        # Save bounding boxes if enabled
        if SAVE_BOUNDING_BOXES:
            save_bounding_boxes(validated_boxes, image_num, is_test)
        
        # Create bounding box visualization if enabled
        if VISUALIZE_BOUNDING_BOXES:
            visualize_bounding_boxes(canvas, validated_boxes, image_num, is_test)
        
        # Verify coordinate consistency (show for first image only)
        if image_num == 0:
            verify_coordinate_consistency(validated_boxes, image_num, is_test)
        
        if (image_num + 1) % 50 == 0 or image_num == NUM_IMAGES - 1:  # Progress every 50 images
            print(f"Generated image {START_NUMBER + image_num}/{START_NUMBER + NUM_IMAGES - 1} ({'test' if is_test else 'train'}) - Progress: {((image_num + 1)/NUM_IMAGES)*100:.1f}%")

if __name__ == "__main__":
    main()
