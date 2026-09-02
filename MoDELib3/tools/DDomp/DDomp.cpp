/* This file is part of MODEL, the Mechanics Of Defect Evolution Library.
 *
 * Copyright (C) 2012 by Giacomo Po <gpo@ucla.edu>
 *
 * MODEL is distributed without any warranty under the
 * GNU General Public License (GPL) v2 <http://www.gnu.org/licenses/>.
 */
// Define the non-singluar method used for calculations
#define _MODEL_NON_SINGULAR_DD_ 1 // 0 classical theory, 1 Cai's regularization method, 2 Lazar's regularization method

#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <DefectiveCrystal.h>

#include <iostream>
#include <string>

using namespace model;

/*! SERVER MODE -- why this exists.
 *
 *  DDomp is a one-shot batch program, but the DisloCluster coupled march uses
 *  it as a repeatedly-called function: once per fast solve, 64 times on a
 *  40 dpa run. Every invocation rebuilds state that is IDENTICAL across all of
 *  them. Profiled at 189 533 CD nodes, of a 398.5 s invocation:
 *
 *      Cholesky of the diffusion operator   175.6 s   44.1%
 *      process startup + WSL launch          22.8 s    5.7%
 *      mesh read + create + faces            12.0 s    3.0%
 *      TrialFunction construction             2.1 s    0.5%
 *      ------------------------------------------------------
 *      repeated per invocation              212.5 s   53.3%
 *
 *  The Cholesky alone is larger than the entire iterative solve (156.3 s,
 *  39.2%) and scales as N^1.86 against the solve's N^1.33, so its share grows
 *  with every increase in domain size. It is performed to serve ONE diffusion
 *  solve of 0.93 s, and the matrix it factorizes -- the diffusion operator --
 *  depends on the mesh and the diffusion tensor, neither of which changes
 *  between fast solves.
 *
 *  In server mode the mesh, the trial functions, the DefectiveCrystal and its
 *  factorization are built ONCE; each cycle re-reads only the configuration.
 *  `ClusterDynamicsFEM::initializeConfiguration` assigns the mobile and
 *  immobile fields and touches nothing else, and `initializeSolver` is guarded
 *  by `solverInitialized`, so the factorization is not repeated -- which is
 *  what makes the answer bit-identical rather than merely close.
 *
 *  PROTOCOL (line-oriented, on stdin/stdout):
 *      SOLVE  -> re-read evl, run the steps, print the DONE sentinel
 *      QUIT   -> exit
 *  Anything else is ignored. Without --server the behaviour is byte-for-byte
 *  the original: read once, solve once, exit.
 */
namespace
{
    const std::string doneSentinel("@@DDOMP_DONE@@");
}

int main (int argc, char* argv[])
{

#ifdef _MODEL_PYBIND11_ // COMPILED WITH PYBIND11
    pybind11::scoped_interpreter guard{};
#endif

    std::string folderName("./");
    bool server(false);
    for(int k=1;k<argc;++k)
    {
        const std::string arg(argv[k]);
        if(arg=="--server")
        {
            server=true;
        }
        else
        {
            folderName=arg;
        }
    }

    try
    {
        DislocationDynamicsBase<3> ddBase(folderName);
        // The step counter is not the only thing runSteps() advances:
        // runSingleStep does `totalTime += dt` and recomputes dt, and
        // ClusterDynamics takes its lastUpdateTime from totalTime. Restoring
        // runID alone left cycle 2 starting from cycle 1's clock, which came
        // out as a 2.3e-9 drift against a fresh process -- small, but this
        // change exists to be exactly equivalent, not nearly.
        const long int startID(ddBase.simulationParameters.runID);
        const double startTime(ddBase.simulationParameters.totalTime);
        const double startDt(ddBase.simulationParameters.dt);
        DefectiveCrystal<3> DC(ddBase);

        {// the first solve is the original path, in full
            DDconfigIO<3> configIO(ddBase.simulationParameters.traitsIO.evlFolder);
            configIO.read(ddBase.simulationParameters.runID);
            DC.initializeConfiguration(configIO);
            DC.runSteps();
        }

        if(server)
        {
            std::cout<<doneSentinel<<std::endl;
            std::string cmd;
            while(std::getline(std::cin,cmd))
            {
                while(cmd.size() && (cmd.back()=='\r' || cmd.back()==' '))
                {
                    cmd.pop_back();
                }
                if(cmd=="QUIT")
                {
                    break;
                }
                if(cmd!="SOLVE")
                {
                    continue;
                }
                // runSteps() advances runID to Nsteps and then stops, so the
                // counter has to go back to where this case starts or the next
                // cycle would do nothing at all -- silently returning the
                // previous answer. The clock goes back with it.
                ddBase.simulationParameters.runID=startID;
                ddBase.simulationParameters.totalTime=startTime;
                ddBase.simulationParameters.dt=startDt;
                DDconfigIO<3> configIO(ddBase.simulationParameters.traitsIO.evlFolder);
                configIO.read(ddBase.simulationParameters.runID);
                DC.initializeConfiguration(configIO);
                DC.runSteps();
                std::cout<<doneSentinel<<std::endl;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
        if(server)
        {// never leave the driver waiting on a sentinel that cannot come
            std::cout<<doneSentinel<<std::endl;
        }
        return 1;
    }

    return 0;
}
