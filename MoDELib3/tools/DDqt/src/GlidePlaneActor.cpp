/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2011 by Giacomo Po <gpo@ucla.edu>.
 *
 * model is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */

#ifndef model_GlidePlaneActor_cpp_
#define model_GlidePlaneActor_cpp_

#include <numbers>
#include <vtkLine.h>
#include <vtkPlanes.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkLookupTable.h>
#include <vtkStructuredGridAppend.h>
#include <vtkTableBasedClipDataSet.h>
#include <vtkTriangle.h>
#include <vtkCellData.h>

#include <GlidePlaneActor.h>
#include <Polygon2D.h>

namespace model
{

    SingleGlidePlaneActor::SingleGlidePlaneActor(const GlidePlane<3>& glidePlane_in) :
    /* init */ glidePlane(glidePlane_in)
    {
    }

    void SingleGlidePlaneActor::appendClosedPolygon(const std::vector<Eigen::Matrix<double,2,1>>& newPoints)
    {
        size_t nextPointID(points.size());
        
        for(size_t k=0;k<newPoints.size();++k)
        {
            
            const auto iterK(uniquePointsIDs.find(newPoints[k]));
            int kID(nextPointID);
            if(iterK!=uniquePointsIDs.end())
            {// newPoints[k] exists
                kID=iterK->second;
            }
            else
            {// newPoints[k] does not exists
                points.emplace_back(newPoints[k]);
                uniquePointsIDs.emplace(newPoints[k],nextPointID);
                nextPointID++;
            }
            
            const size_t k1(k<newPoints.size()-1? k+1 : 0);
            const auto iterK1(uniquePointsIDs.find(newPoints[k1]));
            int k1ID(nextPointID);
            if(iterK1!=uniquePointsIDs.end())
            {// newPoints[k1] exists
                k1ID=iterK1->second;
            }
            else
            {// newPoints[k1] does not exists
                points.emplace_back(newPoints[k1]);
                uniquePointsIDs.emplace(newPoints[k1],nextPointID);
                nextPointID++;
            }
            
            segments.emplace_back((Eigen::Matrix<int,2,1>()<<kID,k1ID).finished());
        }
        
    }

    GlidePlaneActor::GlidePlaneActor(vtkGenericOpenGLRenderWindow* const renWin,vtkRenderer* const ren,DefectiveCrystal<3>& defectiveCrystal_in) :
    /* init */ renderWindow(renWin)
    /* init */,renderer(ren)
    /* init */,defectiveCrystal(defectiveCrystal_in)
    /* init */,dislocationNetwork(defectiveCrystal.template getUniqueTypedMicrostructure<DislocationNetwork<3,0>>())
    /* init */,mainLayout(new QGridLayout(this))
    /* init */,glidePlanesGroup(new QGroupBox(tr("&Planes")))
    /* init */,lut(vtkSmartPointer<vtkLookupTable>::New())
    /* init */,glidePlanesNoiseGroup(new QGroupBox(tr("&Noise")))
    /* init */,noiseMeshSizeLabel(new QLabel("mesh size",this))
    /* init */,noiseMeshSizeEdit(new QLineEdit("2.0"))
    /* init */,grainNoiseLabel(new QLabel("grain",this))
    /* init */,grainNoiseBox(new QComboBox(this))
    /* init */,slipSystemNoiseBox(new QComboBox(this))
    /* init */,glidePlanesNoiseBox(new QComboBox(this))
    /* init */,ssNoiseMin(new QLineEdit("0.0"))
    /* init */,ssNoiseMax(new QLineEdit("0.0"))
    /* init */,sfNoiseMin(new QLineEdit("0.0"))
    /* init */,sfNoiseMax(new QLineEdit("0.0"))
    /* init */,noisePolydata(vtkSmartPointer<vtkPolyData>::New())
    /* init */,noiseMapper(vtkSmartPointer<vtkDataSetMapper>::New())
    /* init */,noiseActor(vtkSmartPointer<vtkActor>::New())
    /* init */,glidePlaneMeshGroup(new QGroupBox(tr("&Stacking Faults")))
    /* init */,glidePlanePolydata(vtkSmartPointer<vtkPolyData>::New())
    /* init */,glidePlaneMapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    /* init */,glidePlaneActor(vtkSmartPointer<vtkActor>::New())
    /* init */,meshPolydata(vtkSmartPointer<vtkPolyData>::New())
    /* init */,meshMapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    /* init */,meshActor(vtkSmartPointer<vtkActor>::New())
    ///* init */,noiseLimits(Eigen::Array<double,2,2>::Zero())
    {
        
        lut->SetHueRange(0.66667, 0.0);
        lut->Build();

        glidePlanesGroup->setCheckable(true);
        glidePlanesGroup->setChecked(false);
        glidePlaneActor->SetVisibility(false);
        
        glidePlanesNoiseGroup->setCheckable(true);
        glidePlanesNoiseGroup->setChecked(false);
        
        glidePlaneMeshGroup->setCheckable(true);
        glidePlaneMeshGroup->setChecked(false);
        meshActor->SetVisibility(false);
        
        for(const auto& pair : defectiveCrystal.ddBase.poly.grains)
        {
            grainNoiseBox->addItem(QString::fromStdString(std::to_string(pair.first)));
        }
        
        glidePlanesNoiseBox->addItem("solidSolution_1");
        glidePlanesNoiseBox->addItem("solidSolution_2");
        glidePlanesNoiseBox->addItem("stackingFault");
        const auto& grain(defectiveCrystal.ddBase.poly.grains.begin()->second);
        for(size_t k=0; k<grain.singleCrystal->slipSystems().size();++k)
        {
            slipSystemNoiseBox->addItem(QString::fromStdString("slipSystem "+ std::to_string(k)));
        }
        
        QGridLayout *noiseLayout = new QGridLayout();
        glidePlanesNoiseGroup->setLayout(noiseLayout);
        noiseLayout->addWidget(noiseMeshSizeLabel,0,0,1,1);
        noiseLayout->addWidget(noiseMeshSizeEdit,0,1,1,1);
        noiseLayout->addWidget(grainNoiseLabel,1,0,1,1);
        noiseLayout->addWidget(grainNoiseBox,1,1,1,1);
        noiseLayout->addWidget(slipSystemNoiseBox,2,0,1,1);
        noiseLayout->addWidget(glidePlanesNoiseBox,2,1,1,1);
        noiseLayout->addWidget(ssNoiseMin,3,0,1,1);
        noiseLayout->addWidget(ssNoiseMax,3,1,1,1);
        noiseLayout->addWidget(sfNoiseMin,4,0,1,1);
        noiseLayout->addWidget(sfNoiseMax,4,1,1,1);
        
        mainLayout->addWidget(glidePlanesGroup,0,0,1,1);
        mainLayout->addWidget(glidePlanesNoiseGroup,1,0,1,1);
        mainLayout->addWidget(glidePlaneMeshGroup,2,0,1,1);
        
        this->setLayout(mainLayout);
        connect(glidePlanesGroup,SIGNAL(toggled(bool)), this, SLOT(modify()));
        connect(glidePlanesNoiseGroup,SIGNAL(toggled(bool)), this, SLOT(modify()));
        connect(glidePlaneMeshGroup,SIGNAL(toggled(bool)), this, SLOT(modify()));
        
        connect(glidePlanesNoiseBox,SIGNAL(currentIndexChanged(int)), this, SLOT(modify()));
        connect(slipSystemNoiseBox,SIGNAL(currentIndexChanged(int)), this, SLOT(computeGlidePlaneNoise()));
        connect(ssNoiseMin,SIGNAL(returnPressed()), this, SLOT(modify()));
        connect(ssNoiseMax,SIGNAL(returnPressed()), this, SLOT(modify()));
        connect(sfNoiseMin,SIGNAL(returnPressed()), this, SLOT(modify()));
        connect(sfNoiseMax,SIGNAL(returnPressed()), this, SLOT(modify()));
        connect(noiseMeshSizeEdit,SIGNAL(returnPressed()), this, SLOT(computeGlidePlaneNoise()));

        // GlidePlane boundaries
        glidePlaneMapper->SetInputData(glidePlanePolydata);
        glidePlaneActor->SetMapper(glidePlaneMapper);
        glidePlaneActor->GetProperty()->SetColor(1.0, 0.0, 1.0); //(R,G,B)
        renderer->AddActor(glidePlaneActor);
        
        // GlidePlane noise
        noisePolydata->Allocate();
        noiseMapper->SetInputData(noisePolydata);
        noiseActor->SetMapper ( noiseMapper );
        noiseActor->GetProperty()->SetOpacity(0.8); //Make the mesh have some transparency.
        renderer->AddActor(noiseActor);

        // Stacking Faults
        meshPolydata->Allocate();
        meshMapper->SetInputData(meshPolydata);
        meshActor->SetMapper ( meshMapper );
        meshActor->GetProperty()->SetOpacity(0.8); //Make the mesh have some transparency.
        renderer->AddActor(meshActor);
    }

    void GlidePlaneActor::updateConfiguration()
    {
        std::cout<<"Updating GlidePlanes..."<<std::flush;
        const auto t0= std::chrono::system_clock::now();
        computeMeshIntersections();
        computeGlidePlaneNoise();
        computeStackingFaults();
        std::cout<<magentaColor<<" ["<<(std::chrono::duration<double>(std::chrono::system_clock::now()-t0)).count()<<" sec]"<<defaultColor<<std::endl;
    }

    void GlidePlaneActor::computeGlidePlaneNoise()
    {
        if(dislocationNetwork && glidePlanesNoiseGroup->isChecked())
        {
            noiseValues.clear();
            for(int k=0;k<3;++k)
            {
                valuesMinMax[k]=std::make_pair(std::numeric_limits<double>::max(),-std::numeric_limits<double>::max());
            }
            vtkSmartPointer<vtkPoints> meshPts(vtkSmartPointer<vtkPoints>::New());
            vtkSmartPointer<vtkCellArray> meshTriangles(vtkSmartPointer<vtkCellArray>::New());
            vtkSmartPointer<vtkUnsignedCharArray> meshColors(vtkSmartPointer<vtkUnsignedCharArray>::New());
            meshColors->SetNumberOfComponents(3);
            
            const size_t selectedGrainID(std::stoi(grainNoiseBox->currentText().toStdString()));
            const auto& grain(defectiveCrystal.ddBase.poly.grain(selectedGrainID));
            const size_t selectedSlipSystemID(slipSystemNoiseBox->currentIndex());
            const auto& slipSystem(grain.singleCrystal->slipSystems()[selectedSlipSystemID]);
            const auto& planeNoise(slipSystem->planeNoise);

  
            size_t nodeIDoffset(0);
            double meshSize;
            std::stringstream ssM(noiseMeshSizeEdit->text().toStdString());
            if(ssM >> meshSize)
            {
                noiseMeshSizeEdit->setStyleSheet("background-color: white");
                
                if(planeNoise)
                {
                    for(const auto& weakGlidePlane : dislocationNetwork->ddBase.glidePlaneFactory.glidePlanes())
                    {
                        if(!weakGlidePlane.second.expired())
                        {
                            const auto glidePlane(weakGlidePlane.second.lock());
                            const auto glidePlaneSlipSystems(glidePlane->slipSystems());
                                                    
                            if(glidePlane->grain.grainID==selectedGrainID && glidePlaneSlipSystems.find(slipSystem)!=glidePlaneSlipSystems.end())
                            {
                                std::deque<Eigen::Matrix<double,2,1>> boundaryPts;
                                std::deque<Eigen::Matrix<double,2,1>> internalPts;
                                for(const auto& bndLine : glidePlane->meshIntersections)
                                {
                                    boundaryPts.push_back(glidePlane->localPosition(bndLine->P0));
                                }
                                TriangularMesh triMesh;
                                triMesh.reMesh(boundaryPts,internalPts,meshSize,"pazq");
                                
                                for(const auto& point2d : triMesh.vertices())
                                {
                                    const auto point3d(glidePlane->globalPosition(point2d));
                                    meshPts->InsertNextPoint(point3d(0),point3d(1),point3d(2));
                                    noiseValues.push_back(planeNoise->gridInterp(point2d));
                                    
                                    valuesMinMax[0]=std::make_pair(std::min(valuesMinMax[0].first,std::get<0>(noiseValues.back())),std::max(valuesMinMax[0].second,std::get<0>(noiseValues.back())));
                                    valuesMinMax[1]=std::make_pair(std::min(valuesMinMax[1].first,std::get<1>(noiseValues.back())),std::max(valuesMinMax[1].second,std::get<1>(noiseValues.back())));
                                    valuesMinMax[2]=std::make_pair(std::min(valuesMinMax[2].first,std::get<2>(noiseValues.back())),std::max(valuesMinMax[2].second,std::get<2>(noiseValues.back())));
                                }
                                
                                for(const auto& tri : triMesh.triangles())
                                {
                                    vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
                                    triangle->GetPointIds()->SetId (0,tri(0)+nodeIDoffset);
                                    triangle->GetPointIds()->SetId (1,tri(1)+nodeIDoffset);
                                    triangle->GetPointIds()->SetId (2,tri(2)+nodeIDoffset);
                                    meshTriangles->InsertNextCell ( triangle );
                                    const auto triColor(Eigen::Matrix<int,1,3>::Random()*255);
                                    meshColors->InsertNextTuple3(triColor(0),triColor(1),triColor(2)); // use this to assig color to each vertex
                                }
                                nodeIDoffset += triMesh.vertices().size();
                            }
                        }
                    }
                }
            }
            else
            {
                noiseMeshSizeEdit->setStyleSheet("background-color: red");
            }
            
            noisePolydata->SetPoints ( meshPts );
            noisePolydata->SetPolys ( meshTriangles );
            noisePolydata->GetCellData()->SetScalars(meshColors);
            noiseMapper->SetScalarModeToUseCellData();
            noisePolydata->Modified();
            modify();
        }
    }

void GlidePlaneActor::computeStackingFaults()
{
    if(dislocationNetwork && glidePlaneMeshGroup->isChecked())
    {
        std::map<LatticePlaneKey<3>,SingleGlidePlaneActor> singleGlidePlaneMap;
        for(auto& loop : dislocationNetwork->loops())
        {
            for(const auto& pair : loop.second.lock()->patches().localPatches())
            {
                const auto& glidePlane(pair.first->glidePlane);
                const auto& key(glidePlane->key);
                const auto glidePlaneIter(singleGlidePlaneMap.find(key));
                if(glidePlaneIter!=singleGlidePlaneMap.end())
                {// glide plane not found
                    glidePlaneIter->second.appendClosedPolygon(pair.second);
                }
                else
                {// glide plane not found
                    const auto success(singleGlidePlaneMap.emplace(std::piecewise_construct,
                                                                   std::forward_as_tuple(key),
                                                                   std::forward_as_tuple(*glidePlane)));
                    if(success.second)
                    {
                        success.first->second.appendClosedPolygon(pair.second);
                    }
                    else
                    {
                        throw std::runtime_error("Cannot insert glide plane in singleGlidePlaneMap");
                    }
                }
            }
        }
        
        vtkSmartPointer<vtkPoints> meshPts(vtkSmartPointer<vtkPoints>::New());
        vtkSmartPointer<vtkCellArray> meshTriangles(vtkSmartPointer<vtkCellArray>::New());
        vtkSmartPointer<vtkUnsignedCharArray> meshColors(vtkSmartPointer<vtkUnsignedCharArray>::New());
        meshColors->SetNumberOfComponents(3);
        
        size_t numVertices(0);
        for(auto& pair : singleGlidePlaneMap)
        {
            TriangularMesh triMesh;
            triMesh.reMesh(pair.second.points,pair.second.segments,100000.0,"pz");
            
            for(const auto& point2d : triMesh.vertices())
            {
                const auto point3d(pair.second.glidePlane.globalPosition(point2d));
                meshPts->InsertNextPoint(point3d(0),point3d(1),point3d(2));
            }
            
            for(const auto& tri : triMesh.triangles())
            {
                vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
                triangle->GetPointIds()->SetId (0,tri(0)+numVertices);
                triangle->GetPointIds()->SetId (1,tri(1)+numVertices);
                triangle->GetPointIds()->SetId (2,tri(2)+numVertices);
                meshTriangles->InsertNextCell ( triangle );
                
                //                    const Eigen::Vector2d triCenter((triMesh.vertices()[tri(0)]+triMesh.vertices()[tri(1)]+triMesh.vertices()[tri(2)])/3.0);
                //                    FINISH HERE, COMPUTE total slip vector and misfit energy
                
                const auto triColor(Eigen::Matrix<int,1,3>::Random()*255);
                meshColors->InsertNextTuple3(triColor(0),triColor(1),triColor(2)); // use this to assig color to each vertex
            }
            numVertices+=triMesh.vertices().size();
        }
        //    std::cout<<"Done meshing"<<std::endl;
        
        meshPolydata->SetPoints ( meshPts );
        meshPolydata->SetPolys ( meshTriangles );
        meshPolydata->GetCellData()->SetScalars(meshColors);
        meshPolydata->Modified();
        meshMapper->SetScalarModeToUseCellData();
        //    renWin->Render();
    }
}

    void GlidePlaneActor::computeMeshIntersections()
    {
        if(dislocationNetwork)
        {
            vtkSmartPointer<vtkPoints> glidePlanePoints = vtkSmartPointer<vtkPoints>::New();
            vtkSmartPointer<vtkCellArray> glidePlaneCells(vtkSmartPointer<vtkCellArray>::New());
            size_t pointIDs(0);
            for(const auto& weakGlidePlane : dislocationNetwork->ddBase.glidePlaneFactory.glidePlanes())
            {
                if(!weakGlidePlane.second.expired())
                {
                    const auto glidePlane(weakGlidePlane.second.lock());
                    const auto& grain(glidePlane->grain);
                    if(glidePlane->slipSystems().size())
                    {
                        
                        // Plot intersections with mesh
                        for(const auto& mshInt : glidePlane->meshIntersections)
                        {
                            glidePlanePoints->InsertNextPoint(mshInt->P0.data());
                            
                            glidePlanePoints->InsertNextPoint(mshInt->P1.data());
                            
                            vtkSmartPointer<vtkLine> line(vtkSmartPointer<vtkLine>::New());
                            line->GetPointIds()->SetId(0, pointIDs); // the second 0 is the index of the Origin in linesPolyData's points
                            line->GetPointIds()->SetId(1, pointIDs+1);
                            glidePlaneCells->InsertNextCell(line);
                            pointIDs+=2;
                        }
                    }
                }
            }
            glidePlanePolydata->SetPoints(glidePlanePoints);
            glidePlanePolydata->SetLines(glidePlaneCells);
            glidePlanePolydata->Modified();
        }
    }

    void GlidePlaneActor::modify()
    {
        glidePlaneActor->SetVisibility(glidePlanesGroup->isChecked());
        meshActor->SetVisibility(glidePlaneMeshGroup->isChecked());
        noiseActor->SetVisibility(glidePlanesNoiseGroup->isChecked());
        
        if(glidePlanesNoiseGroup->isChecked())
        {
            vtkSmartPointer<vtkUnsignedCharArray> noiseColors(vtkSmartPointer<vtkUnsignedCharArray>::New());
            noiseColors->SetNumberOfComponents(3);
            double val(0.0);
            double dclr[3];
            unsigned char cclr[3];
            lut->SetTableRange(valuesMinMax[glidePlanesNoiseBox->currentIndex()].first, valuesMinMax[glidePlanesNoiseBox->currentIndex()].second);
            
            for(const auto& tup : noiseValues)
            {
                switch (glidePlanesNoiseBox->currentIndex())
                {
                    case 0:
                        val=std::get<0>(tup);
                        break;
                    case 1:
                        val=std::get<1>(tup);
                        break;
                    case 2:
                        val=std::get<2>(tup);
                        break;
                    default:
                        break;
                }

                lut->GetColor(val, dclr);
                for(unsigned int j = 0; j < 3; j++)
                {
                    cclr[j] = static_cast<unsigned char>(255.0 * dclr[j]);
                }
                noiseColors->InsertNextTypedTuple(cclr);
            }
            
            noisePolydata->GetPointData()->SetScalars(noiseColors);
            noisePolydata->Modified();
            noiseMapper->SetScalarModeToUsePointData();
        }
        
        renderWindow->Render();
    }

}
#endif
