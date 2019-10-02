import numpy as np

def template_matching(target, template):
    pass

# minSAD = VALUE_MAX;

# // loop through the search image
# for ( size_t x = 0; x <= S_cols - T_cols; x++ ) {
#     for ( size_t y = 0; y <= S_rows - T_rows; y++ ) {
#         SAD = 0.0;

#         // loop through the template image
#         for ( size_t j = 0; j < T_cols; j++ )
#             for ( size_t i = 0; i < T_rows; i++ ) {

#                 pixel p_SearchIMG = S[y+i][x+j];
#                 pixel p_TemplateIMG = T[i][j];
		
#                 SAD += abs( p_SearchIMG.Grey - p_TemplateIMG.Grey );
#             }

#         // save the best found position 
#         if ( minSAD > SAD ) { 
#             minSAD = SAD;
#             // give me min SAD
#             position.bestRow = y;
#             position.bestCol = x;
#             position.bestSAD = SAD;
#         }
#     }
    
# }