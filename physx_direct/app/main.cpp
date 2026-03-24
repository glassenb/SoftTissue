#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "extensions/PxCudaHelpersExt.h"
#include "extensions/PxDeformableVolumeExt.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace physx;
namespace fs = std::filesystem;

namespace
{
struct Options
{
    std::string nodesPath;
    std::string tetsPath;
    std::string fixedPath;
    std::string dumpDir;
    std::string controlFile;
    PxU32 dumpEvery = 0;
    PxU32 steps = 180;
    PxReal dt = 1.0f / 60.0f;
    PxReal young = 12000.0f;
    PxReal poisson = 0.45f;
    PxReal density = 1000.0f;
    PxReal damping = 0.02f;
    bool selfCollision = false;
    PxReal selfCollisionFilterDistance = -1.0f;
    PxReal selfCollisionStressTolerance = 0.9f;
    PxVec3 gravity = PxVec3(0.0f, 0.0f, -9.81f);
};

struct MeshBundle
{
    std::vector<PxVec3> nodes;
    std::vector<PxU32> tets;
    std::vector<PxU32> fixedNodes;
};

struct PhysXContext
{
    PxDefaultAllocator allocator;
    PxDefaultErrorCallback errorCallback;
    PxFoundation* foundation = nullptr;
    PxPhysics* physics = nullptr;
    PxCudaContextManager* cudaContext = nullptr;
    PxDefaultCpuDispatcher* dispatcher = nullptr;
    PxScene* scene = nullptr;
    PxMaterial* rigidMaterial = nullptr;
    PxDeformableVolume* body = nullptr;
    PxDeformableVolumeMesh* bodyMesh = nullptr;
    PxDeformableVolumeMaterial* bodyMaterial = nullptr;
    PxShape* bodyShape = nullptr;
    PxVec4* readbackPositions = nullptr;
    PxVec4* simPosInvMassPinned = nullptr;
    PxVec4* simVelocityPinned = nullptr;
    PxVec4* collPosInvMassPinned = nullptr;
    PxVec4* restPosPinned = nullptr;
    PxVec4* kinematicTargetsPinned = nullptr;
    PxVec4* kinematicTargetsDevice = nullptr;
    std::vector<PxVec3> restPositions;
};

void usage()
{
    std::cerr
        << "Usage:\n"
        << "  st_physx_gm_reference --nodes <nodes_f32.bin> --tets <tets_u32.bin> [--fixed <fixed_nodes_u32.bin>]\n"
        << "                        [--steps 180] [--dt 0.0166667] [--young 12000] [--poisson 0.45]\n"
        << "                        [--density 1000] [--damping 0.02] [--gravity gx gy gz]\n"
        << "                        [--self-collision] [--self-collision-filter-distance 0.003]\n"
        << "                        [--self-collision-stress-tolerance 0.9]\n"
        << "                        [--dump-dir <dir> --dump-every 2]\n"
        << "                        [--control-file <gravity_control.txt>]\n";
}

bool parseArgs(int argc, char** argv, Options& opts)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> const char*
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value after " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--nodes")
            opts.nodesPath = next("--nodes");
        else if (arg == "--tets")
            opts.tetsPath = next("--tets");
        else if (arg == "--fixed")
            opts.fixedPath = next("--fixed");
        else if (arg == "--dump-dir")
            opts.dumpDir = next("--dump-dir");
        else if (arg == "--dump-every")
            opts.dumpEvery = static_cast<PxU32>(std::stoul(next("--dump-every")));
        else if (arg == "--control-file")
            opts.controlFile = next("--control-file");
        else if (arg == "--steps")
            opts.steps = static_cast<PxU32>(std::stoul(next("--steps")));
        else if (arg == "--dt")
            opts.dt = std::stof(next("--dt"));
        else if (arg == "--young")
            opts.young = std::stof(next("--young"));
        else if (arg == "--poisson")
            opts.poisson = std::stof(next("--poisson"));
        else if (arg == "--density")
            opts.density = std::stof(next("--density"));
        else if (arg == "--damping")
            opts.damping = std::stof(next("--damping"));
        else if (arg == "--self-collision")
            opts.selfCollision = true;
        else if (arg == "--self-collision-filter-distance")
            opts.selfCollisionFilterDistance = std::stof(next("--self-collision-filter-distance"));
        else if (arg == "--self-collision-stress-tolerance")
            opts.selfCollisionStressTolerance = std::stof(next("--self-collision-stress-tolerance"));
        else if (arg == "--gravity")
        {
            if (i + 3 >= argc)
            {
                std::cerr << "--gravity requires 3 values\n";
                return false;
            }
            opts.gravity = PxVec3(std::stof(argv[++i]), std::stof(argv[++i]), std::stof(argv[++i]));
        }
        else if (arg == "--help" || arg == "-h")
        {
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    return !opts.nodesPath.empty() && !opts.tetsPath.empty();
}

PxReal estimateSelfCollisionFilterDistance(const MeshBundle& mesh)
{
    PxReal minEdge = std::numeric_limits<PxReal>::max();
    const size_t tetCount = mesh.tets.size() / 4;
    const size_t sampleCount = PxMin<size_t>(tetCount, 4096);
    for (size_t i = 0; i < sampleCount; ++i)
    {
        const PxU32* tet = &mesh.tets[i * 4];
        const PxVec3& a = mesh.nodes[tet[0]];
        const PxVec3& b = mesh.nodes[tet[1]];
        const PxVec3& c = mesh.nodes[tet[2]];
        const PxVec3& d = mesh.nodes[tet[3]];
        const PxReal edges[6] = {
            (a - b).magnitude(),
            (a - c).magnitude(),
            (a - d).magnitude(),
            (b - c).magnitude(),
            (b - d).magnitude(),
            (c - d).magnitude(),
        };
        for (PxReal e : edges)
        {
            if (e > 1.0e-8f)
                minEdge = PxMin(minEdge, e);
        }
    }

    if (!PxIsFinite(minEdge) || minEdge == std::numeric_limits<PxReal>::max())
        return 0.003f;
    return PxClamp(minEdge * 0.5f, 0.001f, 0.01f);
}

template <typename T>
bool loadRawVector(const std::string& path, std::vector<T>& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }

    in.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    if (size % sizeof(T) != 0)
    {
        std::cerr << "File size is not aligned for " << path << "\n";
        return false;
    }

    out.resize(size / sizeof(T));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
    return static_cast<size_t>(in.gcount()) == size;
}

bool loadBundle(const Options& opts, MeshBundle& mesh)
{
    std::vector<float> nodeFloats;
    if (!loadRawVector(opts.nodesPath, nodeFloats) || nodeFloats.size() % 3 != 0)
        return false;
    mesh.nodes.resize(nodeFloats.size() / 3);
    for (size_t i = 0; i < mesh.nodes.size(); ++i)
        mesh.nodes[i] = PxVec3(nodeFloats[i * 3 + 0], nodeFloats[i * 3 + 1], nodeFloats[i * 3 + 2]);

    if (!loadRawVector(opts.tetsPath, mesh.tets) || mesh.tets.size() % 4 != 0)
        return false;

    if (!opts.fixedPath.empty())
    {
        if (!loadRawVector(opts.fixedPath, mesh.fixedNodes))
            return false;
    }

    return true;
}

bool initPhysX(const Options& opts, PhysXContext& px)
{
    px.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, px.allocator, px.errorCallback);
    if (!px.foundation)
        return false;

    PxCudaContextManagerDesc cudaDesc;
    px.cudaContext = PxCreateCudaContextManager(*px.foundation, cudaDesc, PxGetProfilerCallback());
    if (!px.cudaContext || !px.cudaContext->contextIsValid())
    {
        std::cerr << "Failed to initialize PhysX CUDA context.\n";
        return false;
    }

    PxTolerancesScale scale;
    px.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *px.foundation, scale, false, nullptr);
    if (!px.physics)
        return false;

    PxInitExtensions(*px.physics, nullptr);

    PxSceneDesc sceneDesc(px.physics->getTolerancesScale());
    sceneDesc.gravity = opts.gravity;
    sceneDesc.cudaContextManager = px.cudaContext;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    sceneDesc.gpuMaxNumPartitions = 8;
    sceneDesc.solverType = PxSolverType::eTGS;
    px.dispatcher = PxDefaultCpuDispatcherCreate(4);
    sceneDesc.cpuDispatcher = px.dispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    px.scene = px.physics->createScene(sceneDesc);
    if (!px.scene)
        return false;

    px.rigidMaterial = px.physics->createMaterial(0.5f, 0.5f, 0.0f);
    return px.rigidMaterial != nullptr;
}

bool createBody(const Options& opts, const MeshBundle& mesh, PhysXContext& px)
{
    PxCookingParams params(px.physics->getTolerancesScale());
    params.meshWeldTolerance = 0.0f;
    params.meshPreprocessParams = PxMeshPreprocessingFlags();
    params.buildTriangleAdjacencies = false;
    params.buildGPUData = true;

    PxArray<PxVec3> verts;
    verts.resize(static_cast<PxU32>(mesh.nodes.size()));
    for (PxU32 i = 0; i < static_cast<PxU32>(mesh.nodes.size()); ++i)
        verts[i] = mesh.nodes[i];

    PxArray<PxU32> tetIndices;
    tetIndices.resize(static_cast<PxU32>(mesh.tets.size()));
    for (PxU32 i = 0; i < static_cast<PxU32>(mesh.tets.size()); ++i)
        tetIndices[i] = mesh.tets[i];

    PxTetrahedronMeshDesc simDesc(verts, tetIndices);
    PxTetrahedronMeshDesc collDesc(verts, tetIndices);
    PxDeformableVolumeSimulationDataDesc simDataDesc;

    px.bodyMesh = PxCreateDeformableVolumeMesh(params, simDesc, collDesc, simDataDesc, px.physics->getPhysicsInsertionCallback());
    if (!px.bodyMesh)
    {
        std::cerr << "PxCreateDeformableVolumeMesh failed.\n";
        return false;
    }

    px.body = px.physics->createDeformableVolume(*px.cudaContext);
    if (!px.body)
    {
        std::cerr << "createDeformableVolume failed.\n";
        return false;
    }

    px.bodyMaterial = PxGetPhysics().createDeformableVolumeMaterial(opts.young, opts.poisson, 0.1f);
    if (!px.bodyMaterial)
        return false;
    px.bodyMaterial->setDamping(opts.damping);
    px.bodyMaterial->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxTetrahedronMeshGeometry geometry(px.bodyMesh->getCollisionMesh());
    px.bodyShape = px.physics->createShape(geometry, &px.bodyMaterial, 1, true, shapeFlags);
    if (!px.bodyShape)
        return false;

    px.body->attachShape(*px.bodyShape);
    px.body->attachSimulationMesh(*px.bodyMesh->getSimulationMesh(), *px.bodyMesh->getDeformableVolumeAuxData());
    px.scene->addActor(*px.body);

    PxDeformableVolumeExt::allocateAndInitializeHostMirror(
        *px.body,
        px.cudaContext,
        px.simPosInvMassPinned,
        px.simVelocityPinned,
        px.collPosInvMassPinned,
        px.restPosPinned);

    const PxReal maxInvMassRatio = 50.0f;
    PxDeformableVolumeExt::transform(
        *px.body,
        PxTransform(PxIdentity),
        1.0f,
        px.simPosInvMassPinned,
        px.simVelocityPinned,
        px.collPosInvMassPinned,
        px.restPosPinned);
    PxDeformableVolumeExt::updateMass(*px.body, opts.density, maxInvMassRatio, px.simPosInvMassPinned);
    PxDeformableVolumeExt::copyToDevice(
        *px.body,
        PxDeformableVolumeDataFlag::eALL,
        px.simPosInvMassPinned,
        px.simVelocityPinned,
        px.collPosInvMassPinned,
        px.restPosPinned);

    px.body->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !opts.selfCollision);
    if (opts.selfCollision)
    {
        const PxReal filterDistance =
            opts.selfCollisionFilterDistance > 0.0f ? opts.selfCollisionFilterDistance : estimateSelfCollisionFilterDistance(mesh);
        px.body->setSelfCollisionFilterDistance(filterDistance);
        px.body->setSelfCollisionStressTolerance(opts.selfCollisionStressTolerance);
        std::cout << "self_collision=on"
                  << " filter_distance=" << filterDistance
                  << " stress_tolerance=" << opts.selfCollisionStressTolerance
                  << "\n";
    }
    else
    {
        std::cout << "self_collision=off\n";
    }
    px.body->setSolverIterationCounts(40);

    const PxU32 vertexCount = px.body->getCollisionMesh()->getNbVertices();
    px.restPositions.resize(vertexCount);
    for (PxU32 i = 0; i < vertexCount; ++i)
        px.restPositions[i] = PxVec3(px.simPosInvMassPinned[i].x, px.simPosInvMassPinned[i].y, px.simPosInvMassPinned[i].z);

    px.readbackPositions = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *px.cudaContext, vertexCount);

    if (!mesh.fixedNodes.empty())
    {
        std::unordered_set<PxU32> fixed(mesh.fixedNodes.begin(), mesh.fixedNodes.end());
        px.kinematicTargetsPinned = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *px.cudaContext, vertexCount);
        for (PxU32 i = 0; i < vertexCount; ++i)
        {
            const bool isFixed = fixed.find(i) != fixed.end();
            px.kinematicTargetsPinned[i] = PxConfigureDeformableVolumeKinematicTarget(px.simPosInvMassPinned[i], isFixed);
        }

        px.kinematicTargetsDevice = PX_EXT_DEVICE_MEMORY_ALLOC(PxVec4, *px.cudaContext, vertexCount);
        PxScopedCudaLock lock(*px.cudaContext);
        px.cudaContext->getCudaContext()->memcpyHtoD(
            reinterpret_cast<CUdeviceptr>(px.kinematicTargetsDevice),
            px.kinematicTargetsPinned,
            vertexCount * sizeof(PxVec4));
        px.body->setDeformableVolumeFlag(PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC, true);
        px.body->setKinematicTargetBufferD(px.kinematicTargetsDevice);
    }

    return true;
}

void readbackPositions(PhysXContext& px)
{
    const PxU32 vertexCount = px.body->getCollisionMesh()->getNbVertices();
    PxScopedCudaLock lock(*px.cudaContext);
    px.cudaContext->getCudaContext()->memcpyDtoH(
        px.readbackPositions,
        reinterpret_cast<CUdeviceptr>(px.body->getPositionInvMassBufferD()),
        vertexCount * sizeof(PxVec4));
}

void reportStep(PhysXContext& px, PxU32 step)
{
    readbackPositions(px);
    const PxU32 vertexCount = px.body->getCollisionMesh()->getNbVertices();

    PxReal minZ = std::numeric_limits<PxReal>::max();
    PxReal maxZ = -std::numeric_limits<PxReal>::max();
    PxReal maxDisp = 0.0f;
    PxReal meanDisp = 0.0f;
    for (PxU32 i = 0; i < vertexCount; ++i)
    {
        const PxVec3 p(px.readbackPositions[i].x, px.readbackPositions[i].y, px.readbackPositions[i].z);
        minZ = PxMin(minZ, p.z);
        maxZ = PxMax(maxZ, p.z);
        const PxReal disp = (p - px.restPositions[i]).magnitude();
        maxDisp = PxMax(maxDisp, disp);
        meanDisp += disp;
    }
    meanDisp /= PxMax(PxU32(1), vertexCount);

    std::cout << "step=" << step
              << " min_z=" << minZ
              << " max_z=" << maxZ
              << " mean_disp=" << meanDisp
              << " max_disp=" << maxDisp
              << "\n";
}

void ensureDumpMeta(const Options& opts, const MeshBundle& mesh)
{
    if (opts.dumpDir.empty())
        return;
    fs::create_directories(opts.dumpDir);
    std::ofstream out(fs::path(opts.dumpDir) / "dump_meta.json", std::ios::binary);
    out << "{\n"
        << "  \"nodes_path\": \"" << opts.nodesPath << "\",\n"
        << "  \"tets_path\": \"" << opts.tetsPath << "\",\n"
        << "  \"fixed_path\": \"" << opts.fixedPath << "\",\n"
        << "  \"node_count\": " << mesh.nodes.size() << ",\n"
        << "  \"tet_count\": " << (mesh.tets.size() / 4) << ",\n"
        << "  \"steps\": " << opts.steps << ",\n"
        << "  \"dt\": " << opts.dt << ",\n"
        << "  \"dump_every\": " << opts.dumpEvery << "\n"
        << "}\n";
}

void writeBinaryAtomic(const fs::path& path, const void* data, size_t sizeBytes)
{
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeBytes));
    }
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path, ec);
    if (ec)
    {
        fs::copy_file(tmp, path, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp, ec);
    }
}

void writeTextAtomic(const fs::path& path, const std::string& text)
{
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << text;
    }
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path, ec);
    if (ec)
    {
        fs::copy_file(tmp, path, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp, ec);
    }
}

void dumpFrame(PhysXContext& px, const Options& opts, PxU32 frameIndex, PxU32 step)
{
    if (opts.dumpDir.empty())
        return;

    readbackPositions(px);
    const PxU32 vertexCount = px.body->getCollisionMesh()->getNbVertices();
    std::vector<float> xyz(vertexCount * 3);
    for (PxU32 i = 0; i < vertexCount; ++i)
    {
        xyz[i * 3 + 0] = px.readbackPositions[i].x;
        xyz[i * 3 + 1] = px.readbackPositions[i].y;
        xyz[i * 3 + 2] = px.readbackPositions[i].z;
    }

    std::ostringstream name;
    name << "frame_" << std::setw(6) << std::setfill('0') << frameIndex << "_step_" << std::setw(6) << std::setfill('0') << step << ".bin";
    const fs::path framePath = fs::path(opts.dumpDir) / name.str();
    writeBinaryAtomic(framePath, xyz.data(), xyz.size() * sizeof(float));
    writeBinaryAtomic(fs::path(opts.dumpDir) / "latest.bin", xyz.data(), xyz.size() * sizeof(float));
    writeTextAtomic(fs::path(opts.dumpDir) / "latest_step.txt", std::to_string(step) + "\n");
}

bool maybeUpdateGravityControl(const Options& opts, PhysXContext& px, fs::file_time_type& lastWrite, PxVec3& currentGravity)
{
    if (opts.controlFile.empty())
        return false;

    const fs::path path(opts.controlFile);
    std::error_code ec;
    if (!fs::exists(path, ec))
        return false;

    const auto writeTime = fs::last_write_time(path, ec);
    if (ec || writeTime == lastWrite)
        return false;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    in >> gx >> gy >> gz;
    if (!in.good() && !in.eof())
        return false;

    currentGravity = PxVec3(gx, gy, gz);
    px.scene->setGravity(currentGravity);
    lastWrite = writeTime;
    std::cout << "gravity=" << currentGravity.x << "," << currentGravity.y << "," << currentGravity.z << "\n";
    return true;
}

void cleanup(PhysXContext& px)
{
    if (px.kinematicTargetsDevice)
        PX_EXT_DEVICE_MEMORY_FREE(*px.cudaContext, px.kinematicTargetsDevice);
    if (px.kinematicTargetsPinned)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.kinematicTargetsPinned);
    if (px.readbackPositions)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.readbackPositions);
    if (px.simPosInvMassPinned)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.simPosInvMassPinned);
    if (px.simVelocityPinned)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.simVelocityPinned);
    if (px.collPosInvMassPinned)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.collPosInvMassPinned);
    if (px.restPosPinned)
        PX_EXT_PINNED_MEMORY_FREE(*px.cudaContext, px.restPosPinned);

    PX_RELEASE(px.bodyShape);
    PX_RELEASE(px.bodyMaterial);
    PX_RELEASE(px.body);
    PX_RELEASE(px.bodyMesh);
    PX_RELEASE(px.rigidMaterial);
    PX_RELEASE(px.scene);
    PX_RELEASE(px.dispatcher);
    if (px.physics)
        PxCloseExtensions();
    PX_RELEASE(px.physics);
    PX_RELEASE(px.cudaContext);
    PX_RELEASE(px.foundation);
}
} // namespace

int main(int argc, char** argv)
{
    Options opts;
    if (!parseArgs(argc, argv, opts))
    {
        usage();
        return 2;
    }

    MeshBundle mesh;
    if (!loadBundle(opts, mesh))
        return 1;
    ensureDumpMeta(opts, mesh);

    std::cout << "Loaded mesh: nodes=" << mesh.nodes.size()
              << " tets=" << (mesh.tets.size() / 4)
              << " fixed_nodes=" << mesh.fixedNodes.size()
              << "\n";

    PhysXContext px;
    if (!initPhysX(opts, px))
    {
        cleanup(px);
        return 1;
    }

    if (!createBody(opts, mesh, px))
    {
        cleanup(px);
        return 1;
    }

    PxU32 frameIndex = 0;
    fs::file_time_type lastControlWrite = fs::file_time_type::min();
    PxVec3 currentGravity = opts.gravity;
    maybeUpdateGravityControl(opts, px, lastControlWrite, currentGravity);
    reportStep(px, 0);
    if (!opts.dumpDir.empty())
        dumpFrame(px, opts, frameIndex++, 0);

    for (PxU32 step = 1; step <= opts.steps; ++step)
    {
        maybeUpdateGravityControl(opts, px, lastControlWrite, currentGravity);
        px.scene->simulate(opts.dt);
        px.scene->fetchResults(true);
        if (step == 1 || step == opts.steps || step % 30 == 0)
            reportStep(px, step);
        if (!opts.dumpDir.empty() && opts.dumpEvery > 0 && (step % opts.dumpEvery == 0 || step == opts.steps))
            dumpFrame(px, opts, frameIndex++, step);
    }

    cleanup(px);
    return 0;
}
