/* This file is part of MoDELib, the Mechanics Of Defects Evolution Library.
 *
 *
 * MoDELib is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_SutherlandHodgman_H_
#define model_SutherlandHodgman_H_

#include <cfloat>
#include <tuple>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <assert.h>

namespace model
{

    class SutherlandHodgman
    {
        //! Clips a subject polygon with a convex clipping polygon
        
        template<typename Aexpr,typename Bexpr>
        constexpr static double dot(const Aexpr& a, const Bexpr& b)
        {
            return a.dot(b);
        }
        
        template<typename Aexpr,typename Bexpr>
        constexpr static double cross(const Aexpr& a, const Bexpr& b)
        {
            return a(0) * b(1) - b(0) * a(1);
        }
        
        template<typename Pexpr,typename Aexpr,typename Bexpr>
        constexpr static bool is_inside(const Pexpr& point, const Aexpr& a, const Bexpr& b)
        {// check if a point is on the LEFT side of an edge
            return (cross(a - b, point) + cross(b, a)) < 0.0;
        }
        
        template<typename PointType>
        constexpr static PointType intersection(const PointType& a1, const PointType& a2, const PointType& b1, const PointType& b2)
        {
            return ((b1 - b2) * cross(a1, a2) - (a1 - a2) * cross(b1, b2)) *
            (1.0 / cross(a1 - a2, b1 - b2));
        }
        
        
    public:
        
        template<typename PointType>
        constexpr static std::vector<PointType> clip(const std::vector<PointType>& subject_polygon, const std::vector<PointType>& clip_polygon)
        {// Sutherland-Hodgman clipping
            if (clip_polygon.empty() || subject_polygon.empty())
            {
                return std::vector<PointType>();
            }
            
            std::vector<PointType> ring{subject_polygon.begin(), subject_polygon.end()};
            
            PointType p1 = clip_polygon[clip_polygon.size() - 1];
            
            std::vector<PointType> input;
            
            for (PointType p2 : clip_polygon)
            {
                input.clear();
                input.insert(input.end(), ring.begin(), ring.end());
                PointType s = input[input.size() - 1];
                
                ring.clear();
                
                for (PointType e : input)
                {
                    if (is_inside(e, p1, p2))
                    {
                        if (!is_inside(s, p1, p2))
                        {
                            ring.push_back(intersection(p1, p2, s, e));
                        }
                        
                        ring.push_back(e);
                    }
                    else if (is_inside(s, p1, p2))
                    {
                        ring.push_back(intersection(p1, p2, s, e));
                    }
                    
                    s = e;
                }
                
                p1 = p2;
            }
            
            return ring;
            
            
            
            //
            //
            //
            //
            //            //   const int   N = 99; // clipped (new) polygon size
            //            const int   N = subjectPolygon.size()+clipPolygon.size(); // clipped (new) polygon size
            ////            PointType cp1, cp2, s, e, inputPolygon[N], newPolygon[N];
            //            PointType cp1, cp2, s, e;
            //            std::vector<PointType> inputPolygon(N);
            //            std::vector<PointType> newPolygon(N);
            //
            //            // copy subject polygon to new polygon and set its size
            //            for(size_t i = 0; i < subjectPolygon.size(); i++)
            //            {
            //                newPolygon[i] = subjectPolygon[i];
            //            }
            //
            //            size_t newPolygonSize = subjectPolygon.size();
            //
            //            for(size_t j = 0; j < clipPolygon.size(); j++)
            //            {
            //                // copy new polygon to input polygon & set counter to 0
            //                for(size_t k = 0; k < newPolygonSize; k++)
            //                {
            //                    inputPolygon[k] = newPolygon[k];
            //                }
            //                int counter = 0;
            //
            //                // get clipping polygon edge
            //                cp1 = clipPolygon[j];
            //                cp2 = clipPolygon[(j + 1) % clipPolygon.size()];
            //
            //                for(size_t i = 0; i < newPolygonSize; i++)
            //                {
            //                    // get subject polygon edge
            //                    s = inputPolygon[i];
            //                    e = inputPolygon[(i + 1) % newPolygonSize];
            //
            //                    if(inside(s, cp1, cp2) && inside(e, cp1, cp2))
            //                    {// Case 1: Both vertices are inside:
            //                        // Only the second vertex is added to the output list
            //                        newPolygon[counter++] = e;
            //                    }
            //                    else if(!inside(s, cp1, cp2) && inside(e, cp1, cp2))
            //                    {// Case 2: First vertex is outside while second one is inside:
            //                        // Both the point of intersection of the edge with the clip boundary
            //                        // and the second vertex are added to the output list
            //                        newPolygon[counter++] = intersection(cp1, cp2, s, e);
            //                        newPolygon[counter++] = e;
            //                    }
            //                    else if(inside(s, cp1, cp2) && !inside(e, cp1, cp2))
            //                    {// Case 3: First vertex is inside while second one is outside:
            //                        // Only the point of intersection of the edge with the clip boundary
            //                        // is added to the output list
            //                        newPolygon[counter++] = intersection(cp1, cp2, s, e);
            //                    }
            //                    else if(!inside(s, cp1, cp2) && !inside(e, cp1, cp2))
            //                    {// Case 4: Both vertices are outside
            //                        // No vertices are added to the output list
            //                    }
            //                }
            //                // set new polygon size
            //                newPolygonSize = counter;
            //            }
            //            return std::vector<PointType>(&newPolygon[0],&newPolygon[0]+newPolygonSize);
        }
        
    };

} /* namespace model */
#endif
