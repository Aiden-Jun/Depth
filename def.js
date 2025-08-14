/*
Definitions to be used across all other files
Just to be organized

All uppercase variables
*/

// Color defs
export const COLOR = {
    white:0, 
    black:1
}

// Piece defs
export const PIECES = {
    empty: 0,
	wp: 1, wn: 2, wb: 3, wr: 4, wq: 5, wk: 6,
    bp: 7, bn: 8, bb: 9, br: 10, bq: 11, bk: 12
}

// Maps for UCI converting
export const RANKMAP = { 1:"a", 2:"b", 3:"c", 4:"d", 5:"e", 6:"f", 7:"g", 8:"h" }
export const FILEMAP = { 2:"8", 3:"8", 4:"8", 5:"8", 6:"8", 7:"8", 8:"8", 9:"8" }
export const PROMOTIONMAP = {
    [pieces.wq]: "q", [pieces.wr]: "r", [pieces.wn]: "n", [pieces.wb]: "b", 
    [pieces.bq]: "q", [pieces.br]: "r", [pieces.bn]: "n", [pieces.bb]: "b"
}

// Offboard indexes
export const OFFBOARD = [
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    29, 30,
    39, 40,
    49, 50,
    59, 60,
    69, 70,
    79, 80,
    89, 90,
    99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119
]

// Returns the color of the piece
export const GETPIECECOLOR = (piece) => {
	if (piece <= 6) {
		return colors.white
	} else if (piece >= 7) {
		return colors.black
	}
}

// Returns the opposing color
export const GETOPPOSINGCOLOR = (color) => {
	if (color === colors.white) {
		return colors.black
	} else if (color === colors.black) {
		return colors.white
    }
}