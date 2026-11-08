import pygame
import numpy as np
from gomoku_engine import GomokuEngine
from utils import load_cnn_model
from mcts import MCTS

WIDTH, HEIGHT = 600, 600
BOARD_SIZE = 15
CELL_SIZE = WIDTH // BOARD_SIZE
BG_COLOR = (220, 179, 92)
LINE_COLOR = (0, 0, 0)


def ask_rematch(screen, winner):
    font_title = pygame.font.SysFont(None, 32)
    font_btn = pygame.font.SysFont(None, 24)

    box_width, box_height = 320, 120
    box_x, box_y = 10, HEIGHT - box_height - 10
    center_x = box_x + box_width // 2

    box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    yes_rect = pygame.Rect(box_x + 10, box_y + 70, 145, 40)
    no_rect = pygame.Rect(box_x + 165, box_y + 70, 145, 40)

    pygame.draw.rect(screen, (245, 245, 245), box_rect, border_radius=10)
    pygame.draw.rect(screen, (0, 0, 0), box_rect, 3, border_radius=10)

    title_text = f"Winner: {winner}" if winner != 'Draw' else "Draw!"
    title_surf = font_title.render(title_text, True, (0, 0, 0))
    question_surf = font_btn.render("Play a rematch (swap sides)?", True, (50, 50, 50))

    screen.blit(title_surf, title_surf.get_rect(center=(center_x, box_y + 25)))
    screen.blit(question_surf, question_surf.get_rect(center=(center_x, box_y + 50)))

    pygame.draw.rect(screen, (46, 139, 87), yes_rect, border_radius=8)
    pygame.draw.rect(screen, (178, 34, 34), no_rect, border_radius=8)

    yes_surf = font_btn.render("Yes (Y/Enter)", True, (255, 255, 255))
    no_surf = font_btn.render("No (N/Esc)", True, (255, 255, 255))
    screen.blit(yes_surf, yes_surf.get_rect(center=yes_rect.center))
    screen.blit(no_surf, no_surf.get_rect(center=no_rect.center))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_y, pygame.K_RETURN):
                    return True
                if event.key in (pygame.K_n, pygame.K_ESCAPE):
                    return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.collidepoint(event.pos):
                    return True
                if no_rect.collidepoint(event.pos):
                    return False


def draw_board(screen, engine):
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2),
                         (i * CELL_SIZE + CELL_SIZE // 2, HEIGHT - CELL_SIZE // 2), 2)
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                         (WIDTH - CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), 2)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = engine.board[r][c]
            center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
            if stone == 'X':
                pygame.draw.circle(screen, (0, 0, 0), center, CELL_SIZE // 2 - 4)
            elif stone == 'O':
                pygame.draw.circle(screen, (255, 255, 255), center, CELL_SIZE // 2 - 4)
    pygame.display.flip()


def play_with_gui(path: str, structure, num_simulations: int = 1000, c_puct: float = 1.5, human_starts: bool = True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gomoku: Human vs AI")

    model = load_cnn_model(path, structure)

    playing = True
    while playing:
        engine = GomokuEngine()

        mcts = MCTS(model, num_simulations=num_simulations, c_puct=c_puct)

        if human_starts:
            human_player = 'X'
            bot_player = 'O'
        else:
            human_player = 'O'
            bot_player = 'X'

        draw_board(screen, engine)

        while not engine.is_game_over:

            # Tura bota
            if engine.current_player == bot_player:
                pygame.event.pump()
                pi = mcts.get_action_probs(engine, temperature=0.0)
                action = int(np.argmax(pi))
                r, c = action // engine.board_size, action % engine.board_size

                engine.make_move(r, c)
                mcts.update_with_move(action)

                draw_board(screen, engine)
                if engine.is_game_over:
                    print(f"Game Finished! Winner: {engine.winner}")
                    break

            else:
                move_made = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                    if event.type == pygame.MOUSEBUTTONDOWN and not engine.is_game_over:
                        x, y = event.pos
                        col = x // CELL_SIZE
                        row = y // CELL_SIZE

                        if 0 <= row < engine.board_size and 0 <= col < engine.board_size and engine.board[row][col] == '.':
                            engine.make_move(row, col)

                            action = row * engine.board_size + col
                            mcts.update_with_move(action)

                            draw_board(screen, engine)
                            move_made = True

                            if engine.is_game_over:
                                print(f"Game Finished! Winner: {engine.winner}")
                                break

                if move_made:
                    continue

        if ask_rematch(screen, engine.winner):
            human_starts = not human_starts
        else:
            playing = False

    pygame.quit()