import numpy as np
import pygame
import sys
import tkinter as tk
from tkinter import simpledialog


ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE = 100 
RADIUS = SQUARE_SIZE // 2 - 5
ANIMATION_SPEED = 15


BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


pygame.init()


width = COLUMN_COUNT * SQUARE_SIZE
height = (ROW_COUNT + 1) * SQUARE_SIZE
size = (width, height)
screen = pygame.display.set_mode(size)



def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)



def draw_board(board):
    screen.fill(BLACK)
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


def is_valid_move(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for row in range(ROW_COUNT):
        if board[row][col] == 0:
            return row
    return None


def animate_piece(board, col, piece):
    row = get_next_open_row(board, col)
    if row is None:
        return

    x_pos = col * SQUARE_SIZE + SQUARE_SIZE // 2
    y_pos = SQUARE_SIZE // 2
    final_y_pos = height - (row * SQUARE_SIZE) - SQUARE_SIZE // 2

    while y_pos < final_y_pos:
        screen.fill(BLACK)
        draw_board(board)

        
        pygame.draw.circle(screen, RED if piece == 1 else YELLOW, (x_pos, y_pos), RADIUS)

        pygame.display.update()
        y_pos += ANIMATION_SPEED
        pygame.time.delay(5)

    board[row][col] = piece
    draw_board(board)


def check_win(board, piece):
    
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row][col + i] == piece for i in range(4)):
                return True


    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT):
            if all(board[row + i][col] == piece for i in range(4)):
                return True


    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row + i][col + i] == piece for i in range(4)):
                return True

    for row in range(3, ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if all(board[row - i][col + i] == piece for i in range(4)):
                return True

    return False


def evaluate_window(window, piece):
    opponent_piece = 1 if piece == 2 else 2
    score = 0

    if window.count(piece) == 4:
        score += 100  
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5  
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2  

    if window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 4

    return score


def evaluate_board(board, piece):
    opponent_piece = 1 if piece == 2 else 2
    score = 0

    
    center_array = [board[row][COLUMN_COUNT // 2] for row in range(ROW_COUNT)]
    score += center_array.count(piece) * 3  # Give extra points for center plays

    
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
        self.column = column
        self.score = score
        self.children = []
        self.depth = depth  
        self.pruned=pruned

    def add_child(self, child_node):
        self.children.append(child_node)

    def visualize(self, indent=0):
        """ Recursively prints the tree structure """
        prefix = "  " * indent
        if self.pruned:
            node_info = f"[PRUNED] Node(Column: {self.column}, Score: {self.score})"
        else:
            node_info = f"Node(Column: {self.column}, Score: {self.score})"
        print(prefix + node_info)

        for child in self.children:
            child.visualize(indent + 1)


def minimax(board, depth, maximizingPlayer, level=0):
    valid_columns = [col for col in range(COLUMN_COUNT) if is_valid_move(board, col)]
    current_node = MinimaxNode(column=None, depth=level)

    
    if check_win(board, 2): 
        return MinimaxNode(column=None, score=100000, depth=level)
    elif check_win(board, 1):  
        return MinimaxNode(column=None, score=-100000, depth=level)
    elif len(valid_columns) == 0:  
        return MinimaxNode(column=None, score=0, depth=level)
    elif depth == 0:  
        return MinimaxNode(column=None, score=evaluate_board(board, 2), depth=level)

    if maximizingPlayer:
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

    else: 
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
    current_node = MinimaxNode(column=None, depth=level) 

    
    if check_win(board, 2): 
        return MinimaxNode(column=None, score=100000, depth=level)
    elif check_win(board, 1):  
        return MinimaxNode(column=None, score=-100000, depth=level)
    elif len(valid_columns) == 0:  
        return MinimaxNode(column=None, score=0, depth=level)
    elif depth == 0:  
        return MinimaxNode(column=None, score=evaluate_board(board, 2), depth=level)

    if maximizingPlayer:  
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

            alpha = max(alpha, best_score)  

            if alpha >= beta:
               
                pruned_node = MinimaxNode(column=col, score=None, depth=level + 1, pruned=True)
                current_node.add_child(pruned_node)
                break 

        current_node.column = best_col
        current_node.score = best_score
        return current_node

    else:
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

            beta = min(beta, best_score)

            if beta <= alpha:
                
                pruned_node = MinimaxNode(column=col, score=None, depth=level + 1, pruned=True)
                current_node.add_child(pruned_node)
                break  

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
                if event.key == pygame.K_r:
                    return True
                if event.key == pygame.K_q:
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
        turn = 0  

        draw_board(board)

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                
                if turn == 0 and event.type == pygame.MOUSEBUTTONDOWN:
                    x_pos = event.pos[0]
                    col = x_pos // SQUARE_SIZE

                    if is_valid_move(board, col):
                        animate_piece(board, col, 1)
                        if check_win(board, 1):
                            pygame.time.delay(3000)
                            if show_game_over_screen("Player"):
                                break
                            else:
                                return
                        turn = 1

            
            if turn == 1 and not game_over:
                pygame.time.delay(1000)

                if ai_mode == "minimax":
                    tree = minimax(board, 4, True)
                    chosen_col = tree.column
                elif ai_mode == "alpha_beta":
                    tree = minimax_alpha_beta(board, 4, -float("inf"), float("inf"), True)
                    chosen_col = tree.column
                elif ai_mode == "expected_minimax":
                    tree = minimax(board, 4, True)
                    best_col = tree.column  

                    
                    possible_moves = [best_col]
                    probabilities = [0.6]

                    
                    if best_col > 0 and is_valid_move(board, best_col - 1):
                        possible_moves.append(best_col - 1)
                        probabilities.append(0.2)

                    
                    if best_col < COLUMN_COUNT - 1 and is_valid_move(board, best_col + 1):
                        possible_moves.append(best_col + 1)
                        probabilities.append(0.2)

                  
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()

                    
                    chosen_col = np.random.choice(possible_moves, p=probabilities)
                else:
                    raise ValueError("Invalid AI mode selected.")

                print("\nMinimax Decision Tree:")
                tree.visualize()

                
                if is_valid_move(board, chosen_col):
                    animate_piece(board, chosen_col, 2)
                    if check_win(board, 2):
                        pygame.time.delay(3000)
                        if show_game_over_screen("AI"):
                            break
                        else:
                            return
                    turn = 0 




if __name__ == "__main__":
    main()
