import numpy as np
import pygame
import sys
import tkinter as tk
from tkinter import simpledialog

# Constants for game dimensions
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE = 100  # Size of each cell
RADIUS = SQUARE_SIZE // 2 - 5
ANIMATION_SPEED = 15  # Speed of the falling animation

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()

# Set screen size
width = COLUMN_COUNT * SQUARE_SIZE
height = (ROW_COUNT + 1) * SQUARE_SIZE  # Extra row for UI interaction
size = (width, height)
screen = pygame.display.set_mode(size)


# Create board using NumPy
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)


# Draw the game board
def draw_board(board):
    screen.fill(BLACK)  # Clear screen
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            pygame.draw.rect(screen, BLUE, (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK,
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2),
                               RADIUS)

    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            if board[row][col] == 1:
                pygame.draw.circle(screen, RED, (
                    col * SQUARE_SIZE + SQUARE_SIZE // 2, height - (row * SQUARE_SIZE) - SQUARE_SIZE // 2), RADIUS)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, YELLOW, (
                    col * SQUARE_SIZE + SQUARE_SIZE // 2, height - (row * SQUARE_SIZE) - SQUARE_SIZE // 2), RADIUS)

    pygame.display.update()


# Check if a column has space for a move
def is_valid_move(board, col):
    return board[ROW_COUNT - 1][col] == 0  # Check top row


# Find the lowest available row in a column
def get_next_open_row(board, col):
    for row in range(ROW_COUNT):
        if board[row][col] == 0:
            return row
    return None


# Animate the falling piece
def animate_piece(board, col, piece):
    row = get_next_open_row(board, col)
    if row is None:
        return  # No space in the column

    x_pos = col * SQUARE_SIZE + SQUARE_SIZE // 2
    y_pos = SQUARE_SIZE // 2  # Start from the top
    final_y_pos = height - (row * SQUARE_SIZE) - SQUARE_SIZE // 2

    while y_pos < final_y_pos:
        screen.fill(BLACK)  # Clear screen before drawing
        draw_board(board)  # Draw static pieces

        # Draw the falling piece
        pygame.draw.circle(screen, RED if piece == 1 else YELLOW, (x_pos, y_pos), RADIUS)

        pygame.display.update()
        y_pos += ANIMATION_SPEED  # Move down by animation speed
        pygame.time.delay(5)  # Small delay for smoother animation

    board[row][col] = piece  # Place piece in final position
    draw_board(board)  # Redraw the board


def check_win(board, piece):
    # Check horizontal locations
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row][col + i] == piece for i in range(4)):
                return True

    # Check vertical locations
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT):
            if all(board[row + i][col] == piece for i in range(4)):
                return True

    # Check positively sloped diagonals
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row + i][col + i] == piece for i in range(4)):
                return True

    # Check negatively sloped diagonals
    for row in range(3, ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row - i][col + i] == piece for i in range(4)):
                return True

    return False


def evaluate_window(window, piece):
    opponent_piece = 1 if piece == 2 else 2
    score = 0

    if window.count(piece) == 4:
        score += 100  # Winning move
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5  # Almost winning move
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2  # Decent position

    if window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 4  # Block opponent’s almost winning move

    return score


def evaluate_board(board, piece):
    opponent_piece = 1 if piece == 2 else 2
    score = 0

    # Center column preference (AI should play near the center)
    center_array = [board[row][COLUMN_COUNT // 2] for row in range(ROW_COUNT)]
    score += center_array.count(piece) * 3  # Give extra points for center plays

    # Score horizontal, vertical, and diagonal sequences
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row][col + i] for i in range(4)]
            score += evaluate_window(window, piece)

    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT):
            window = [board[row + i][col] for i in range(4)]
            score += evaluate_window(window, piece)

    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += evaluate_window(window, piece)

    for row in range(3, ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row - i][col + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def drop_piece(board, col, piece):
    row = get_next_open_row(board, col)
    if row is not None:
        board[row][col] = piece


class MinimaxNode:
    def __init__(self, column, score=None, depth=0, pruned = False):
        self.column = column  # Column chosen at this node
        self.score = score  # Score at this node (if leaf)
        self.children = []  # Subnodes (possible future moves)
        self.depth = depth  # Tree depth (for formatting)
        self.pruned=pruned

    def add_child(self, child_node):
        self.children.append(child_node)

    def visualize(self, indent=0):
        """ Recursively prints the tree structure """
        prefix = "  " * indent  # Indentation for hierarchy
        if self.pruned:
            node_info = f"[PRUNED] Node(Column: {self.column}, Score: {self.score})"
        else:
            node_info = f"Node(Column: {self.column}, Score: {self.score})"
        print(prefix + node_info)

        for child in self.children:
            child.visualize(indent + 1)  # Recursive call with more indent


def minimax(board, depth, maximizingPlayer, level=0):
    valid_columns = [col for col in range(COLUMN_COUNT) if is_valid_move(board, col)]
    current_node = MinimaxNode(column=None, depth=level)  # Root node

    # Base case: check for terminal state (win/loss/draw)
    if check_win(board, 2):  # AI wins
        return MinimaxNode(column=None, score=100000, depth=level)
    elif check_win(board, 1):  # Player wins
        return MinimaxNode(column=None, score=-100000, depth=level)
    elif len(valid_columns) == 0:  # No more moves (draw)
        return MinimaxNode(column=None, score=0, depth=level)
    elif depth == 0:  # Depth limit reached
        return MinimaxNode(column=None, score=evaluate_board(board, 2), depth=level)

    if maximizingPlayer:  # AI's turn (maximize)
        best_score = -float("inf")
        best_col = valid_columns[0]

        for col in valid_columns:
            temp_board = board.copy()
            drop_piece(temp_board, col, 2)

            child_node = minimax(temp_board, depth - 1, False, level + 1)
            child_node.column = col
            current_node.add_child(child_node)

            if child_node.score > best_score:
                best_score = child_node.score
                best_col = col

        current_node.column = best_col
        current_node.score = best_score
        return current_node

    else:  # Human's turn (minimize)
        best_score = float("inf")
        best_col = valid_columns[0]

        for col in valid_columns:
            temp_board = board.copy()
            drop_piece(temp_board, col, 1)

            child_node = minimax(temp_board, depth - 1, True, level + 1)
            child_node.column = col
            current_node.add_child(child_node)

            if child_node.score < best_score:
                best_score = child_node.score
                best_col = col

        current_node.column = best_col
        current_node.score = best_score
        return current_node


def minimax_alpha_beta(board, depth, alpha, beta, maximizingPlayer, level=0):
    valid_columns = [col for col in range(COLUMN_COUNT) if is_valid_move(board, col)]
    current_node = MinimaxNode(column=None, depth=level)  # Root node

    # Base case: check for terminal state (win/loss/draw)
    if check_win(board, 2):  # AI wins
        return MinimaxNode(column=None, score=100000, depth=level)
    elif check_win(board, 1):  # Player wins
        return MinimaxNode(column=None, score=-100000, depth=level)
    elif len(valid_columns) == 0:  # No more moves (draw)
        return MinimaxNode(column=None, score=0, depth=level)
    elif depth == 0:  # Depth limit reached
        return MinimaxNode(column=None, score=evaluate_board(board, 2), depth=level)

    if maximizingPlayer:  # AI's turn (maximize)
        best_score = -float("inf")
        best_col = valid_columns[0]

        for col in valid_columns:
            temp_board = board.copy()
            drop_piece(temp_board, col, 2)

            child_node = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False, level + 1)
            child_node.column = col
            current_node.add_child(child_node)

            if child_node.score > best_score:
                best_score = child_node.score
                best_col = col

            alpha = max(alpha, best_score)  # Update alpha

            if alpha >= beta:
                # **Mark this node as pruned**
                pruned_node = MinimaxNode(column=col, score=None, depth=level + 1, pruned=True)
                current_node.add_child(pruned_node)
                break  # Prune remaining branches

        current_node.column = best_col
        current_node.score = best_score
        return current_node

    else:  # Human's turn (minimize)
        best_score = float("inf")
        best_col = valid_columns[0]

        for col in valid_columns:
            temp_board = board.copy()
            drop_piece(temp_board, col, 1)

            child_node = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True, level + 1)
            child_node.column = col
            current_node.add_child(child_node)

            if child_node.score < best_score:
                best_score = child_node.score
                best_col = col

            beta = min(beta, best_score)  # Update beta

            if beta <= alpha:
                # **Mark this node as pruned**
                pruned_node = MinimaxNode(column=col, score=None, depth=level + 1, pruned=True)
                current_node.add_child(pruned_node)
                break  # Prune remaining branches

        current_node.column = best_col
        current_node.score = best_score
        return current_node



def show_game_over_screen(winner):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 50)
    text = f"{winner} Wins! Press R to Restart or Q to Quit"
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(width // 2, height // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart the game
                    return True
                if event.key == pygame.K_q:  # Quit the game
                    pygame.quit()
                    sys.exit()
        pygame.time.delay(100)


def choose_ai_mode():
    print("Choose AI Mode:")
    print("1 - Minimax")
    print("2 - Minimax with Alpha-Beta Pruning")
    print("3 - Expected Minimax")

    while True:
        try:
            choice = int(input("Enter 1, 2 or 3: "))
            if choice == 1:
                return "minimax"
            elif choice == 2:
                return "alpha_beta"
            elif choice == 3:
                return "expected_minimax"
            else:
                print("Invalid input. Please enter 1, 2 or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    ai_mode = choose_ai_mode()
    while True:
        board = create_board()
        game_over = False
        turn = 0  # 0 = Human, 1 = AI

        draw_board(board)

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Human player input
                if turn == 0 and event.type == pygame.MOUSEBUTTONDOWN:
                    x_pos = event.pos[0]  # Get x-coordinate of click
                    col = x_pos // SQUARE_SIZE  # Determine which column was clicked

                    if is_valid_move(board, col):
                        animate_piece(board, col, 1)  # Animate piece drop for Player
                        if check_win(board, 1):
                            pygame.time.delay(3000)
                            if show_game_over_screen("Player"):
                                break  # Restart game
                            else:
                                return  # Quit game
                        turn = 1  # Switch to AI

            # AI player move
            if turn == 1 and not game_over:
                pygame.time.delay(1000)  # Small delay for realism

                if ai_mode == "minimax":
                    tree = minimax(board, 4, True)
                    chosen_col = tree.column  # Best move from Minimax
                elif ai_mode == "alpha_beta":
                    tree = minimax_alpha_beta(board, 4, -float("inf"), float("inf"), True)
                    chosen_col = tree.column  # Best move from Alpha-Beta
                elif ai_mode == "expected_minimax":
                    tree = minimax(board, 4, True)  # Use normal Minimax first
                    best_col = tree.column  # Best move from Minimax

                    # Possible moves with probabilities
                    possible_moves = [best_col]  # 60% probability
                    probabilities = [0.6]

                    # Check if left move is valid
                    if best_col > 0 and is_valid_move(board, best_col - 1):
                        possible_moves.append(best_col - 1)
                        probabilities.append(0.2)

                    # Check if right move is valid
                    if best_col < COLUMN_COUNT - 1 and is_valid_move(board, best_col + 1):
                        possible_moves.append(best_col + 1)
                        probabilities.append(0.2)

                    # Normalize probabilities (if only 2 moves are possible)
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()

                    # Select column based on probability
                    chosen_col = np.random.choice(possible_moves, p=probabilities)
                else:
                    raise ValueError("Invalid AI mode selected.")

                print("\nMinimax Decision Tree:")
                tree.visualize()

                # Drop AI piece in chosen column
                if is_valid_move(board, chosen_col):
                    animate_piece(board, chosen_col, 2)  # Animate AI move
                    if check_win(board, 2):
                        pygame.time.delay(3000)
                        if show_game_over_screen("AI"):
                            break  # Restart game
                        else:
                            return  # Quit game
                    turn = 0  # Switch back to Human


  # Print the tree structure



# Run the game


if __name__ == "__main__":
    main()
