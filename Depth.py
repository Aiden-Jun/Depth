import time
import random

# Definitions
colors = {
    "white":0, "black":1,
    0: "white", 1: "black"
}

pieces = {
    "e":0,
    "wp": 1, "wn": 2, "wb": 3, "wr": 4, "wq": 5, "wk": 6,
    "bp": 7, "bn": 8, "bb": 9, "br": 10, "bq": 11, "bk": 12
}

pvm = {
    pieces["wp"]: 100, pieces["wn"]: 290, pieces["wb"]: 320, pieces["wr"]: 500, pieces["wq"]: 920,
    pieces["bp"]: -100, pieces["bn"]: -290, pieces["bb"]: -320, pieces["br"]: -500, pieces["bq"]: -920,
}

pim = {
    pieces["e"]: " ",
    pieces["wp"]: "♙", pieces["wn"]: "♘", pieces["wb"]: "♗", pieces["wr"]: "♖", pieces["wq"]: "♕", pieces["wk"]: "♔",
    pieces["bp"]: "♟", pieces["bn"]: "♞", pieces["bb"]: "♝", pieces["br"]: "♜", pieces["bq"]: "♛", pieces["bk"]: "♚",
}


pst = {
    "p": (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    "n": ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    "b": ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    "r": (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    "q": (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    "k": (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

piece_to_pst = {
    1: 'p', 7: 'p',   # pawns
    2: 'n', 8: 'n',   # knights
    3: 'b', 9: 'b',   # bishops
    4: 'r', 10: 'r',  # rooks
    5: 'q', 11: 'q',  # queens
    6: 'k', 12: 'k',  # kings
}

pst_strength = 0.7

rank_map = {2:"a", 3:"b", 4:"c", 5:"d", 6:"e", 7:"f", 8:"g", 9:"h"}
file_map = {2:"8", 3:"7", 4:"6", 5:"5", 6:"4", 7:"3", 8:"2", 9:"1"}

fen_piece_map = {
    'P': pieces["wp"], 'N': pieces["wn"], 'B': pieces["wb"], 'R': pieces["wr"], 'Q': pieces["wq"], 'K': pieces["wk"],
    'p': pieces["bp"], 'n': pieces["bn"], 'b': pieces["bb"], 'r': pieces["br"], 'q': pieces["bq"], 'k': pieces["bk"]
}

ob = []
for i in range(120):
    file = i % 10
    rank = i // 10
    if file < 2 or file > 9 or rank < 2 or rank > 9:
        ob.append(i)

def gpc(piece):
    if piece == pieces["e"]:
        return None
    elif piece <= 6:
        return colors["white"]
    elif piece >= 7:
        return colors["black"]
    
def goc(color):
    if color == colors["white"]:
        return colors["black"]
    elif color == colors["black"]:
        return colors["white"]
    
def index120_to_64(index120):
    rank = index120 // 10 - 2
    file = index120 % 10 - 2
    return rank * 8 + file

def index_to_pos(index):
    return rank_map[int(str(index)[1])] + file_map[int(str(index)[0])]

def cut_board_array(board_array):
    cut_board = []
    for index, sq in enumerate(board_array):
        if index not in ob:
            cut_board.append(sq)
    return cut_board

def make_move(board, move):
    new_board_array = board.board_array[:]
    sp = move.sp
    dp = move.dp

    # Move piece
    new_board_array[dp] = new_board_array[sp]
    new_board_array[sp] = pieces["e"]

    # Handle promotion
    if move.promo is not None:
        new_board_array[dp] = move.promo

    # Handle castling
    castling_rights = board.castling_rights.copy()
    if new_board_array[dp] in [pieces["wk"], pieces["bk"]]:
        # Remove castling rights for that side
        if new_board_array[dp] == pieces["wk"]:
            castling_rights["wks"] = False
            castling_rights["wqs"] = False
        else:
            castling_rights["bks"] = False
            castling_rights["bqs"] = False

        # Move the rook if castling
        if abs(sp - dp) == 2:
            if dp > sp:  # kingside
                rook_sp = sp + 3
                rook_dp = sp + 1
            else:  # queenside
                rook_sp = sp - 4
                rook_dp = sp - 1
            new_board_array[rook_dp] = new_board_array[rook_sp]
            new_board_array[rook_sp] = pieces["e"]

    # Switch turn
    new_board_color = goc(board.current_color)

    return BoardState(
        new_board_array,
        current_color=new_board_color,
        castling_rights=castling_rights,
        en_pas_t=None
    )

def default_board():
        board_array = [pieces["e"]] * 120

        # White pieces
        board_array[92] = pieces["wr"]
        board_array[93] = pieces["wn"]
        board_array[94] = pieces["wb"]
        board_array[95] = pieces["wq"]
        board_array[96] = pieces["wk"]
        board_array[97] = pieces["wb"]
        board_array[98] = pieces["wn"]
        board_array[99] = pieces["wr"]
        for file in range(2, 10):
            board_array[82 + file - 2] = pieces["wp"]

        # Black pieces
        board_array[22] = pieces["br"]
        board_array[23] = pieces["bn"]
        board_array[24] = pieces["bb"]
        board_array[25] = pieces["bq"]
        board_array[26] = pieces["bk"]
        board_array[27] = pieces["bb"]
        board_array[28] = pieces["bn"]
        board_array[29] = pieces["br"]
        for file in range(2, 10):
            board_array[32 + file - 2] = pieces["bp"]

        return BoardState(
            board_array,
            current_color=colors["white"],
            castling_rights={"wks": True, "wqs": True, "bks": True, "bqs": True},
            en_pas_t=None
        )

def random_position(move_count):
    logics = Logics()
    board = default_board()
    
    for i in range(move_count):
        legal_moves = logics.legal_moves(board)
        random_move = random.choice(legal_moves)
        
        board = make_move(board, random_move)
        
    return board

def fen_string_to_board(fen):
    board_array = [pieces["e"]] * 120
    parts = fen.split()
    board_part = parts[0]
    turn_part = parts[1]
    castling_part = parts[2]
    en_passant_part = parts[3]
    ranks = board_part.split('/')

    if len(ranks) != 8:
        raise ValueError("Invalid FEN")

    for r, rank in enumerate(ranks):
        file = 0
        for c in rank:
            if c.isdigit():
                file += int(c)
            else:
                index = (2 + r) * 10 + (2 + file)
                board_array[index] = fen_piece_map[c]
                file += 1

    current_color = colors["white"] if turn_part == 'w' else colors["black"]

    castling_rights = {"wks": False, "wqs": False, "bks": False, "bqs": False}
    if 'K' in castling_part:
        castling_rights["wks"] = True
    if 'Q' in castling_part:
        castling_rights["wqs"] = True
    if 'k' in castling_part:
        castling_rights["bks"] = True
    if 'q' in castling_part:
        castling_rights["bqs"] = True

    if en_passant_part != '-':
        file = ord(en_passant_part[0]) - ord('a') + 2
        rank = 10 - int(en_passant_part[1])
        en_pas_t = rank * 10 + file
    else:
        en_pas_t = None
    return BoardState(
        board_array,
        current_color=current_color,
        castling_rights=castling_rights,
        en_pas_t=en_pas_t
    )

class Move:
    def __init__(self, sp, dp, promo, is_capture):
        self.sp = sp
        self.dp = dp
        self.promo = promo
        self.is_capture = is_capture
        
    def get_uci(self):
        if self.promo == None:
            promo = ""
        else:
            promo = self.promo

        return index_to_pos(self.sp) + index_to_pos(self.dp) + promo
    
class BoardState:
    def __init__(self, board_array, current_color, castling_rights, en_pas_t):
        self.board_array = board_array
        self.current_color = current_color
        self.en_pas_t = en_pas_t

        if castling_rights == None:
            self.castling_rights = {"wks":False, "wqs":False, "bks":False, "bqs":False}
        else:
            self.castling_rights = castling_rights

class Logics:
    def pawn_moves(self, color, board, index, only_capture=False):
        moves = []
        board_array = board.board_array
        
        if color != gpc(board_array[index]):
            return []

        # Pawn offsets
        if color == colors["white"]:
            single_move = -10
            double_move = -20
            capture_left = -11
            capture_right = -9
            pawn_start = 8
            pawn_end = 2
            promo_pieces = [pieces["wq"], pieces["wr"], pieces["wb"], pieces["wn"]]
        else:
            single_move = 10
            double_move = 20
            capture_left = 9
            capture_right = 11
            pawn_start = 3
            pawn_end = 9
            promo_pieces = [pieces["bq"], pieces["br"], pieces["bb"], pieces["bn"]]
    
        # Capture left
        if board_array[index + capture_left] != pieces["e"] and gpc(board_array[index + capture_left]) == goc(color):
            target_rank = int((index + capture_left) / 10)
            if target_rank == pawn_end:
                for promo in promo_pieces:
                    moves.append(Move(index, index + capture_left, promo, True))
            else:
                moves.append(Move(index, index + capture_left, None, True))

        # Capture right
        if board_array[index + capture_right] != pieces["e"] and gpc(board_array[index + capture_right]) == goc(color):
            target_rank = int((index + capture_right) / 10)
            if target_rank == pawn_end:
                for promo in promo_pieces:
                    moves.append(Move(index, index + capture_right, promo, True))
            else:
                moves.append(Move(index, index + capture_right, None, True))

        # En passant
        en_pas_t = board.en_pas_t
        if en_pas_t is not None:
            if index + capture_left == en_pas_t:
                moves.append(Move(index, index + capture_left, None, True))
            if index + capture_right == en_pas_t:
                moves.append(Move(index, index + capture_right, None, True))

        if only_capture == False:
            # Double move
            if int(index / 10) == pawn_start:
                if board_array[index + single_move] == pieces["e"] and board_array[index + double_move] == pieces["e"]:
                    moves.append(Move(index, index + double_move, None, False))

            # Single move
            if board_array[index + single_move] == pieces["e"]:
                target_rank = int((index + single_move) / 10)
                if target_rank == pawn_end:
                    for promo in promo_pieces:
                        moves.append(Move(index, index + single_move, promo, False))
                else:
                    moves.append(Move(index, index + single_move, None, False))

        # Filter out off board moves
        moves = [m for m in moves if m.dp not in ob]
        return moves
    
    def knight_moves(self, color, board, index):
        board_array = board.board_array

        offsets = [-17, -15, -10, -6, 6, 10, 15, 17]

        moves = []
        row, col = divmod(index, 8)

        for offset in offsets:
            new_index = index + offset
            if 0 <= new_index < 64:
                new_row, new_col = divmod(new_index, 8)

                if abs(new_row - row) + abs(new_col - col) == 3 and abs(new_row - row) <= 2 and abs(new_col - col) <= 2:
                    target_piece = board_array[new_index]

                    if target_piece == pieces["e"]:
                        if gpc(target_piece) != goc(color):
                            moves.append(Move(index, new_index, None, True))
                        else:
                            moves.append(Move(index, new_index, None, False))
        return moves
    
    def bishop_moves(self, color, board, index):
        moves = []

        board_array = board.board_array

        if color != gpc(board_array[index]):
            return []
        
        bishop_offsets = [-11, -9, 9, 11]

        for offset in bishop_offsets:
            destination = index + offset
            
            while destination not in ob:
                if board_array[destination] == pieces["e"]:
                    moves.append(Move(index, destination, None, False))
                
                # Piece blocking
                if gpc(board_array[destination]) == goc(color):
                    moves.append(Move(index, destination, None, True))
                    break
                elif gpc(board_array[destination]) == color:
                    break

                destination += offset

        return moves
    
    def rook_moves(self, color, board, index):
        moves = []
        
        board_array = board.board_array

        if color != gpc(board_array[index]):
            return []
        
        rook_offsets = [-10, 10, -1, 1]

        for offset in rook_offsets:
            destination = index + offset
            
            while destination not in ob:
                if board_array[destination] == pieces["e"]:
                    moves.append(Move(index, destination, None, False))
                
                # When there is a piece blocking
                if gpc(board_array[destination]) == goc(color):
                    moves.append(Move(index, destination, None, True))
                    break
                elif gpc(board_array[destination]) == color:
                    break

                destination += offset
        
        return moves
    
    def queen_moves(self, color, board, index):
        moves = []
        moves.extend(self.rook_moves(color, board, index))
        moves.extend(self.bishop_moves(color, board, index))
        return moves
    
    def is_index_attacked(self, index, board, attacker_color):
        board_array = board.board_array

        for sq_index, piece in enumerate(board_array):
            if sq_index in ob or piece == pieces["e"]:
                continue

            if gpc(piece) != attacker_color:
                continue

            if piece in [pieces["wp"], pieces["bp"]]:
                moves = self.pawn_moves(attacker_color, board, sq_index, only_capture=True)
            elif piece in [pieces["wn"], pieces["bn"]]:
                moves = self.knight_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wb"], pieces["bb"]]:
                moves = self.bishop_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wr"], pieces["br"]]:
                moves = self.rook_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wq"], pieces["bq"]]:
                moves = self.queen_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wk"], pieces["bk"]]:
                king_offsets = [-10, -9, -1, 1, 9, 10, 11, -11]
                moves = []
                for offset in king_offsets:
                    target = sq_index + offset
                    if target not in ob:
                        moves.append(Move(sq_index, target, None, False))
            else:
                moves = []

            for move in moves:
                if move.dp == index:
                    return True

        return False
    
    def king_moves(self, color, board, index):
        moves = []

        board_array = board.board_array
        
        if color != gpc(board_array[index]):
            return []
        
        castling_rights = board.castling_rights

        king_offsets = [-10, -9, -1, 1, 9, 10, 11, -11]

        for offset in king_offsets:
            if board_array[index + offset] == pieces["e"] or gpc(board_array[index + offset]) == goc(color):
                if index + offset not in ob:
                    if gpc(board_array[index + offset]) == goc(color):
                        moves.append(Move(index, index + offset, None, True))
                    else:
                        moves.append(Move(index, index + offset, None, False))

        if color == colors["white"]:
            # White kingside
            if not self.is_index_attacked(96, board, colors["black"]) and not self.is_index_attacked(98, board, colors["black"]) and castling_rights["wks"] and index == 96 and board_array[index + 1] == pieces["e"] and board_array[index + 2] == pieces["e"] and board_array[index + 3] == pieces["wr"]:
                moves.append(Move(index, index + 2, None, False))

            # White queenside
            if not self.is_index_attacked(96, board, colors["black"]) and not self.is_index_attacked(94, board, colors["black"]) and castling_rights["wqs"] and index == 96 and board_array[index - 1] == pieces["e"] and board_array[index - 2] == pieces["e"] and board_array[index -3] == pieces["e"] and board_array[index - 4] == pieces["wr"]:
                moves.append(Move(index, index - 2, None, False))
        elif color == colors["black"]:
            # Black kingside
            if not self.is_index_attacked(26, board, colors["white"]) and not self.is_index_attacked(28, board, colors["white"]) and castling_rights["bks"] and index == 26 and board_array[index + 1] == pieces["e"] and board_array[index + 2] == pieces["e"] and board_array[index + 3] == pieces["br"]:
                moves.append(Move(index, index + 2, None, False))

            # Black queenside
            if not self.is_index_attacked(26, board, colors["white"]) and not self.is_index_attacked(24, board, colors["white"]) and castling_rights["bqs"] and index == 26 and board_array[index - 1] == pieces["e"] and board_array[index - 2] == pieces["e"] and board_array[index - 3] == pieces["e"] and board_array[index - 4] == pieces["br"]:
                moves.append(Move(index, index - 2, None, False))

        # Filter moves that put king in attack
        moves = [m for m in moves if not self.is_index_attacked(m.dp, board, goc(color))]
        return moves

    def legal_moves(self, board):
        board_array = board.board_array
        current_color = board.current_color

        # Find king position
        king_square = None
        for i, p in enumerate(board_array):
            if current_color == colors["white"] and p == pieces["wk"]:
                king_square = i
                break
            elif current_color == colors["black"] and p == pieces["bk"]:
                king_square = i
                break

        # Generate all pseudo-legal moves
        pseudo_moves = []
        for sq_index, piece in enumerate(board_array):
            if sq_index in ob or piece == pieces["e"]:
                continue
            if gpc(piece) != current_color:
                continue

            if piece in [pieces["wp"], pieces["bp"]]:
                pseudo_moves.extend(self.pawn_moves(current_color, board, sq_index))
            elif piece in [pieces["wn"], pieces["bn"]]:
                pseudo_moves.extend(self.knight_moves(current_color, board, sq_index))
            elif piece in [pieces["wb"], pieces["bb"]]:
                pseudo_moves.extend(self.bishop_moves(current_color, board, sq_index))
            elif piece in [pieces["wr"], pieces["br"]]:
                pseudo_moves.extend(self.rook_moves(current_color, board, sq_index))
            elif piece in [pieces["wq"], pieces["bq"]]:
                pseudo_moves.extend(self.queen_moves(current_color, board, sq_index))
            elif piece in [pieces["wk"], pieces["bk"]]:
                pseudo_moves.extend(self.king_moves(current_color, board, sq_index))

        # Filter out moves that leave king in check
        legal_moves = []
        for move in pseudo_moves:
            new_board = make_move(board, move)

            # Update king square if the king moved
            new_king_square = king_square
            if move.sp == king_square:
                new_king_square = move.dp

            if not self.is_index_attacked(new_king_square, new_board, goc(current_color)):
                legal_moves.append(move)
        
        # Filter out off board moves
        legal_moves = [m for m in legal_moves if m.sp not in ob]
        legal_moves = [m for m in legal_moves if m.dp not in ob]
        return legal_moves
    
class Engine:
    def __init__(self):
        self.logics = Logics()
        self.show_thought = True

    def is_checkmate(self, board):
        return self.logics.legal_moves(board) == []
    
    def eval(self, board):
        eval_score = 0
        
        for sq64 in range(64):
            piece = cut_board_array(board.board_array)[sq64]

            if piece == pieces["e"]:
                continue

            ptype = piece_to_pst[piece]

            if piece not in [pieces["wk"], pieces["bk"]]:
                eval_score += pvm[piece]
                if gpc(piece) == colors["white"]:
                    eval_score += int(pst[ptype][sq64] * pst_strength)
                elif gpc(piece) == colors["black"]:
                    # Flip the pst
                    flipped = [0] * 64
                    for sq in range(64):
                        rank = sq // 8
                        file = sq % 8
                        flipped_sq = (7 - rank) * 8 + file
                        flipped[sq] = pst[ptype][flipped_sq]
                    
                    eval_score -= int(flipped[sq64] * pst_strength)

        return eval_score
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_checkmate(board):
            return self.eval(board), None

        legal_moves = self.logics.legal_moves(board)
        if not legal_moves:
            return self.eval(board), None

        best_move = None

        if maximizing_player:
            max_eval = float("-inf")
            for move in legal_moves:
                new_board = make_move(board, move)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, False)

                if self.show_thought:
                    print(f"\r[Depth {depth}] Trying {move.get_uci()} -> Eval: {eval_score} \033[K", end="")

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if self.show_thought:
                        print(f"\r[Depth {depth}] Beta cutoff at move {move.get_uci()} \033[K", end="")
                    break  # beta cut off
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in legal_moves:
                new_board = make_move(board, move)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, True)

                if self.show_thought:
                    print(f"\r[Depth {depth}] Trying {move.get_uci()} -> Eval: {eval_score} \033[K", end="")

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if self.show_thought:
                        print(f"\r[Depth {depth}] Alpha cutoff at move {move.get_uci()} \033[K", end="")
                    break  # alpha cut off
            return min_eval, best_move
        
    def search_best_move(self, for_color, board, depth):
        print("\033[?25l", end="")

        _, best_move = self.minimax(
            board,
            depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=(for_color == colors["white"])
        )

        print("\033[?25h", end="")

        return best_move

class CLI:
    def __init__(self):
        self.version = "1.0.0"

        self.play_as = colors["white"]

        self.engine = Engine()

        # Settings
        self.search_depth = 3
        self.show_thought = True
        self.show_legal_moves = False
        self.show_eval = True

    def give_options(self, options, returner=None):
        for key, option in options.items():
            print(f"({key}) {option[0]}")
        
        while True:
            chosen = input("> ")
            chosen = chosen.replace(" ", "").lower()

            if chosen not in options:
                print("\n\033[31mOption not found\033[0m\n")
            else:
                print()
                options[chosen][1]()

                # Go back to returner
                if returner != None:
                    print()
                    returner()

                return chosen

    def start(self):
        print("\x1b[44m \033[1mDepth\033[0m\x1b[44m - Python Chess Engine \x1b[0m")
        print(f"\u001b[46m Version: {self.version} \u001b[0m")

        self.give_options({
            "a":("About", self.about),
            "c":("Configure", self.configure),
            "n":("Analyze", self.analyze),
            "p":("Play", self.play),
            "q":("Quit", quit)
        }, returner=self.start)

    def about(self):
        print("\u001b[41m About \u001b[0m")
        print("Depth is a chess engine that was originally going to be written in Javascript.")
        print("However, I used Python because I was more experienced with it.")
        print("It will be slow since it is python.")

    def configure(self, return_to_play=False):
        print("\u001b[42m Configure \u001b[0m")
        def change_search_depth():
            new_depth = input("New depth > ")
            self.search_depth = int(new_depth)
        
        def change_show_thought():
            show_thougth = input("t/f > ")
            if show_thougth.lower() == "t":
                self.show_thought = True
            else:
                self.show_thought = False

        def change_show_legal_moves():
            show_legal_moves = input("t/f > ")
            if show_legal_moves.lower() == "t":
                self.show_legal_moves = True
            else:
                self.show_legal_moves = False

        self.give_options({
                "1": (f"Search Depth: {self.search_depth}", change_search_depth),
                "2": (f"Show thought: {self.show_thought}", change_show_thought),
                "3": (f"Show legal moves: {self.show_legal_moves}", change_show_legal_moves),
                "e": ("Exit", self.start)
        }, returner=self.configure)

    def analyze(self):
        print("\u001b[45m Analyze \u001b[0m")
        print("G to generate random board")
        print("Enter FEN string to analyze (E to exit)")
        fen_string = input("> ")
            
        if fen_string.lower() == "e":
            print()
            self.start()

        if fen_string.lower() == "g":
            print()
            print("Move count <= 100")
            move_count = int(input("> "))
            if move_count <= 100:
                board = random_position(move_count)
            else:
                return
        else:
            board = fen_string_to_board(fen_string)

        print()
        self.print_board(board)

        # Count captured pieces
        starting_counts = {
            pieces["wp"]: 8, pieces["wn"]: 2, pieces["wb"]: 2, pieces["wr"]: 2, pieces["wq"]: 1,
            pieces["bp"]: 8, pieces["bn"]: 2, pieces["bb"]: 2, pieces["br"]: 2, pieces["bq"]: 1
        }

        current_counts = {p:0 for p in starting_counts}

        for sq in board.board_array:
            if sq in current_counts:
                current_counts[sq] += 1

        captured_white = []
        captured_black = []

        for piece, start_count in starting_counts.items():
            captured = start_count - current_counts[piece]
            if captured > 0:
                if piece <= 6:  # white piece
                    captured_black.extend([piece] * captured)
                else:            # black piece
                    captured_white.extend([piece] * captured)

        piece_icon_map = {
            pieces["wp"]: "♙", pieces["wn"]: "♘", pieces["wb"]: "♗", pieces["wr"]: "♖", pieces["wq"]: "♕", pieces["wk"]: "♔",
            pieces["bp"]: "♟", pieces["bn"]: "♞", pieces["bb"]: "♝", pieces["br"]: "♜", pieces["bq"]: "♛", pieces["bk"]: "♚",
        }

        print()
        print("Captured by White: ", " ".join(piece_icon_map[p] for p in captured_white) or "None")
        print("Captured by Black: ", " ".join(piece_icon_map[p] for p in captured_black) or "None")

        # Evaluate position
        eval_score = self.engine.eval(board)
        print(f"Evaluation: {eval_score}")

        # Evaluation bar
        max_bar_length = 20
        scale = 100
        bar_length = int(min(abs(eval_score) / scale, max_bar_length))

        if eval_score > 0:
            bar = "█" * bar_length + "-" * (max_bar_length - bar_length)
            print(f"[{bar}] White is better")
        elif eval_score < 0:
            bar = "█" * bar_length + "-" * (max_bar_length - bar_length)
            print(f"[{bar}] Black is better")
        else:
            bar = "-" * max_bar_length
            print(f"[{bar}] Equal")

        print()
        color = board.current_color

        start_time = time.time()
        print(f"Search depth: {self.search_depth}")
        best_move = self.engine.search_best_move(color, board, self.search_depth)
        elapsed = time.time() - start_time
        print()
        print(f"Best for {colors[color]}: {best_move.get_uci()} | Thought for {elapsed:.2f}s")

    def print_board(self, board):
        board_array = board.board_array

        piece_icon_map = {
            pieces["wp"]: "♙", pieces["wn"]: "♘", pieces["wb"]: "♗", pieces["wr"]: "♖", pieces["wq"]: "♕", pieces["wk"]: "♔",
            pieces["bp"]: "♟", pieces["bn"]: "♞", pieces["bb"]: "♝", pieces["br"]: "♜", pieces["bq"]: "♛", pieces["bk"]: "♚",
            pieces["e"]: "."
        }

        if self.play_as == colors["white"]:
            print("  a b c d e f g h")
            for rank in range(2, 10):  # ranks 8 → 1
                row = []
                for file in range(2, 10):  # files a → h
                    index = rank * 10 + file
                    row.append(piece_icon_map.get(board_array[index], "?"))
                print(f"{10 - rank} " + " ".join(row) + f" {10 - rank}")
            print("  a b c d e f g h")

        elif self.play_as == colors["black"]:
            print("  h g f e d c b a")  # flipped files
            for rank in range(9, 1, -1):  # ranks 1 → 8
                row = []
                for file in range(9, 1, -1):  # files h → a
                    index = rank * 10 + file
                    row.append(piece_icon_map.get(board_array[index], "?"))
                print(f"{rank - 1} " + " ".join(row) + f" {rank - 1}")
            print("  h g f e d c b a")

    def play(self):
        print("\u001b[43m Play \u001b[0m")
        print(f"\u001b[1mCurrent configuration\u001b[0m")
        print(f"Search depth: {self.search_depth}")
        print("E to exit\n")
        
        def set_white():
            self.play_as = colors["white"]
            
        def set_black():
            self.play_as = colors["black"]

        self.give_options({
            "1":("Play as white", set_white),
            "2":("Play as black", set_black),
            "e":("Exit", self.start)
        })
        print(f"\u001b[43m Playing as {colors[self.play_as].upper()} \u001b[0m")

        current_board = default_board()

        while True:
            print()
            self.print_board(current_board)

            if current_board.current_color == self.play_as:
                legal_moves = self.engine.logics.legal_moves(current_board)
                if not legal_moves:
                    print("\u001b[41m Checkmate or stalemate, game over \u001b[0m")
                    break
                
                if self.show_eval:
                    print()
                    print(f"Evaluation: {self.engine.eval(current_board)}")
                
                if self.show_legal_moves:
                    for move in legal_moves:
                        print(move.get_uci())
                print(f"{len(legal_moves)} Legal moves found\n")

                move_input = input("\u001b[41mUCI\u001b[0m > ").strip().lower()
                
                if move_input == "e":
                    print()
                    self.start()
                
                move_found = None
                for move in legal_moves:
                    if move.get_uci() == move_input:
                        move_found = move
                        break
                
                if move_found:
                    current_board = make_move(current_board, move_found)
                else:
                    print("\n\033[31mInvalid move, try again\033[0m")
            else:
                print()
                print(f"Evaluation: {self.engine.eval(current_board)}")
                print("Engine thinking...")
                
                start_time = time.time()
                best_move = self.engine.search_best_move(current_board.current_color, current_board, self.search_depth)
                elapsed = time.time() - start_time
                if best_move is None:
                    print("\u001b[41m Engine has no moves, game over \u001b[0m")
                    break
                print("\n")
                print(f"\x1b[44m Engine plays: {best_move.get_uci()} \x1b[0m | Thought for {elapsed:.2f}s")
                current_board = make_move(current_board, best_move)

if __name__ == "__main__":
    app = CLI()
    app.start()