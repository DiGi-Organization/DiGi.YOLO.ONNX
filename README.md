# DiGi.YOLO.ONNX

**DiGi.YOLO.ONNX** scores the frozen YOLO detector in process, from an ONNX export, with no Python
interpreter involved. It is the in-process counterpart of `DiGi.YOLO`'s CPython runner and writes
the same bounding box result file, so everything downstream reads one path exactly as it reads the
other.

---

## 🏗️ Project Architecture & Assemblies

The repository contains the following core components and assemblies:
* **[DiGi.YOLO.ONNX](DiGi.YOLO.ONNX)** (Path: `DiGi.YOLO.ONNX\DiGi.YOLO.ONNX`)

It is a companion of [DiGi.YOLO](https://github.com/DiGi-Organization/DiGi.YOLO) rather than part of
it, so that `DiGi.YOLO` keeps its dependency set — `DiGi.Core` and `System.Text.Json` — and only a
host that actually wants in-process inference pays for ONNX Runtime and OpenCV.

---

## 🤖 What it does

```
image files ──► Emgu.CV decode + letterbox ──► ONNX Runtime ──► NMS ──► .bbrf
```

| | |
|---|---|
| Inference | ONNX Runtime, through `Microsoft.ML.OnnxRuntime.Managed` |
| Decoding, resizing, padding | Emgu.CV 4.12.0.5764 |
| Output | the same `.bbrf` contract `DiGi.YOLO` writes and reads |

**Emgu.CV is not an arbitrary imaging choice.** It is the same OpenCV that ultralytics calls through
`cv2` — the same JPEG decoder and the same bilinear resize — so the two paths differ only in the
arithmetic of the graph, not in what was fed into it. That is what makes the agreement measured
below reachable at all.

**ONNX Runtime rather than OpenCV's own DNN module**, which Emgu also exposes and which would have
cost no extra package: `Emgu.CV.runtime.windows.cuda` is published on nuget.org only up to
`4.4.0.4099`, and the workspace is on `4.12`. OpenCV DNN through Emgu is therefore CPU-only for the
foreseeable future, while ONNX Runtime reaches a GPU by swapping one native package.

### The native execution provider is the host's choice

This library declares the **managed** bindings only. Each deployed host declares the native provider
it wants, at the same version — `Microsoft.ML.OnnxRuntime` for CPU, `Microsoft.ML.OnnxRuntime.DirectML`
for any DirectX 12 GPU without a CUDA install, or `Microsoft.ML.OnnxRuntime.Gpu` for CUDA — along
with `Emgu.CV` and `Emgu.CV.runtime.windows`. A `HintPath` reference does not carry NuGet packages
with it, and the symptom of forgetting is a partial result rather than an error.

---

## 📦 Producing the model

The ONNX file is a **one-off preparation step that still needs ultralytics**, so removing the Python
dependency removes it from every run rather than from the repository. `DiGi.YOLO`'s `export.py`
produces it, and must be run with the ultralytics version pinned in `DiGi.YOLO`'s `requirements.txt`
(`8.4.165`, verified in [DiGi.YOLO#17](https://github.com/DiGi-Organization/DiGi.YOLO/issues/17) to
reproduce the frozen checkpoint's detections) — a version nobody has checked can export a different
detector. The export is repeated whenever the checkpoint is
retrained: a new checkpoint is a new detector, so the new graph is re-measured against the CPython
path before the in-process numbers are trusted again.

```bash
python export.py --model "user files/YOLO/models/model.pt" --output "user files/YOLO/models/model.onnx"
```

**The graph has to be the raw one-to-many head** — `[batch, 4 + classes, anchors]`, suppressed on the
C# side exactly as ultralytics suppresses it. Since ultralytics 8.4, exporting with `nms=False`
selects the NMS-free one-to-one head of a model that has one (YOLO26), which answers
`[batch, max_det, 6]` and is a different detector from the one the CPython path scores, so the export
leaves `nms` at its default ([DiGi.YOLO#20](https://github.com/DiGi-Organization/DiGi.YOLO/issues/20)).
`Modify.Predict` refuses such a graph before scoring anything — by the `end2end` flag ultralytics
writes into the metadata, or by a declared output that states no more anchors than channels — rather
than decoding it into detections that look plausible and mean nothing.

The provenance of the artefact it produces is recorded in the
[DiGi.YOLO README](https://github.com/DiGi-Organization/DiGi.YOLO#readme), beside the checkpoint's own.

---

## ✅ Measured agreement with the CPython path

The detector is frozen, so this path is not allowed to be a slightly different detector. Agreement
is measured rather than assumed, by `DiGi.Test/DiGi.YOLO.ONNX.xUnit`'s `Predict_Parity` fact: both
runners are pointed at one directory of held images and their two `.bbrf` files are compared
detection by detection.

### The stated tolerance

Observed over the same 2 000 held images of county 104106 for each checkpoint. `train8` is the frozen
detector in production. `train9_fresh` is the retrained detector selected by the gate of
[DiGi.GIS.YOLO.UI#12](https://github.com/DiGi-Organization/DiGi.GIS.YOLO.UI/issues/12), exported on
2026-10-07 and staged beside it. It becomes `model.onnx` when it ships
([#2](https://github.com/DiGi-Organization/DiGi.YOLO.ONNX/issues/2)).

| Bound | Stated | `train8` (YOLOv8x, 2026-09-03) | `train9_fresh` (YOLO26x, 2026-10-07) |
|---|---|---|---|
| Images agreeing on detection count | ≥ 99.5 % | **100 %** (2 000 of 2 000) | **100 %** (2 000 of 2 000) |
| Confident detections with no partner | 0 | **0** of 1 639 | **0** of 1 633 |
| 99th percentile box deviation | ≤ 0.1 px | **0.034 px** (median 0.004 px) | **0.047 px** (median 0.005 px) |
| Box deviations over 1 px (worst) | not bounded | 2 (3.3 px) | 2 (1.9 px) |
| Overlap of any matched pair | ≥ 0.95 | **0.970** | **0.977** |
| Confidence deviation | ≤ 0.01 | **0.001062** | **0.001859** |

The `train8` row was re-measured on 2026-09-30 under ultralytics 8.4.165, the version that exports and
runs both checkpoints. It agreed with the figures above (99th percentile 0.033 px, worst overlap
0.970, worst confidence deviation 0.001394), so a difference between the two columns comes from the
weights, not the toolchain.

The tolerance is not zero and cannot be: one path runs an fp32 CUDA graph through torch and the
other an fp32 graph through ONNX Runtime, so the last bits of every number differ. A detection whose
confidence sits on the reporting threshold legitimately appears on one side and not the other, so
detections inside a guard band around the threshold are left out of the count comparison rather than
counted as disagreements.

**Coordinates are bounded at a percentile rather than at a maximum, on purpose.** Both checkpoints
show the same tail. 2 of about 1 600 matched detections move by more than a pixel: the worst by 3.3 px
(`train8`) and 1.9 px (`train9_fresh`). The worst `train8` case had its confidence agreeing to 1e-5,
with exactly one detection on each side. It is therefore not a suppression tie, and with a median of
0.004 px it is not a systematic shift either.

For YOLOv8 a mechanism fits, though nobody has proven it by instrumenting the graph. Each box edge is
a softmax expectation over sixteen bins (DFL) in units of the feature stride. That expectation is far
more sensitive to the last bits than the single sigmoid behind the class score, so an edge can move a
fraction of a bin, which is several pixels at stride 32, while the score stays put. YOLO26 has no DFL
(`reg_max = 1`, one regressed distance per edge) and still shows the tail, only shorter. So DFL
amplifies the effect but does not cause it alone; a stride-scaled box regression magnifies the last
bits whichever head produces it.

Bounding the maximum would mean stating a tolerance of about 4 px, and a 4 px bound would no longer
notice a real coordinate regression against a median of 0.004 px. Instead, every individual detection
is guarded by the overlap of its matched pair, which stays high however the edges wander.

### Throughput, for the record

The CPython path reaches **12.8 ms/image** on CUDA in steady state. The ONNX path on the **CPU**
execution provider takes **177 ms/image** for `train8` (68 million parameters at 640×640) and
**172 ms/image** for `train9_fresh` (YOLO26x, 56 million). That is expected rather than surprising,
and it is why the pipeline stays on CPython. Choosing a GPU execution provider to make the in-process
path fast ([#1](https://github.com/DiGi-Organization/DiGi.YOLO.ONNX/issues/1)) was closed as not
planned on 2026-10-07: the in-process path is kept as a parity reference, not as a pipeline engine.

## 💻 Coding Guidelines for Developers & AI Agents

All developers and AI agents follow these always-applicable rules (Visual Studio 2026, C#). Task-specific detail lives in
the guidelines listed in §9 — read the one matching the task.

### 1. General coding standards
1. **English only** — identifiers and comments.
2. **No `var`** unless the compiler forces it; target-typed `new()` (`PointNode pointNode = new();`, avoids IDE0090) and
   collection expressions (`List<int> numbers = [];`, avoids IDE0028).
3. **Naming:** camelCase starting with the type name, `_`-suffixed qualifier when needed (`PointNode pointNode_Base`); an
   acronym type lowercases only its first letter (`gISPostgreSQLConverterManager`). Collections: pluralized element type
   (`filterConditions`, `point3Ds` — not `listConditions`). A property named after its unique, descriptive type
   (`public AggregateFunction AggregateFunction { get; set; }`). Primitives: plain camelCase.
4. **Zero warnings/analyzer messages.**
5. **Block-scoped namespaces** (`namespace DiGi.Domain { ... }`); file-scoped are prohibited (`.editorconfig`). The template
   pins `LangVersion` 12; most repositories use `latest`, core libraries target `netstandard2.0` — use
   `System.Threading.Lock` only in net9+ projects at C# 13+.
6. **Parameter line breaks (`<= 7` rule, `DIGI0001`):** ≤ 7 parameters on one line however long; ≥ 8 one per line. Line
   length is never the trigger. Call sites prefer one line; multi-line for complex expressions, lambdas, initializers.
   ```csharp
   public void Calculate(double centerX, double centerY, double radius, double? storeyHeight = null)
   ```
7. **Async method names end with `Async`.**
8. **`CancellationToken` is always the LAST parameter (CA1068)** — every method and overload; add new optional parameters
   before it and reorder the `<param>` tags. At call sites **name every argument after a skipped default**
   (`commandTimeout: 30`, `cancellationToken: cancellationToken`) so an inserted parameter fails to compile instead of
   rebinding. Detect: `grep -rnE "CancellationToken [a-zA-Z_]+( = default)?, " --include=*.cs .`
   ```csharp
   public static async Task<bool> ClearAsync(NpgsqlConnection? npgsqlConnection, string tableName, int commandTimeout = 30, CancellationToken cancellationToken = default)
   ```
9. **Simplify member access (IDE0002) — then rebuild.** C# binds the first segment innermost-outwards with no fallback:
   from `DiGi.GIS.PostgreSQL.UI.Classes`, `WebAPI` is `DiGi.GIS.WebAPI`, so `DiGi.WebAPI.Query.GetAsync` keeps its prefix.
   When same-named extension methods can both apply, call the intended one fully qualified — the `using` decides silently.
10. **Separate projects**, not one monolithic solution.
11. **Output optimization:** no filler; only the necessary code, logic or explanation.
12. **Temporary code carries `TODO [MarkerName]:`** — a PascalCase name of the **migration**, reused at every site so one
    `grep` is the deletion checklist, stating the observable **removal condition** and the tracking issue. A workaround
    for another DiGi repository's defect is temporary too (removal condition = the upstream fix). Never tag permanent code.
13. **`ToString()` is not a key** — identity strings render through `CultureInfo.InvariantCulture` (`"R"` with `-0.0`
    folded, `G29` for `decimal`); `pl-PL` files `[0,5, 12,25]` and nothing throws. Reference: `DiGi.Typology.Visual.Query.Key`.
14. **A member resolved by name through reflection is a contract on the whole hierarchy** — document it at both ends,
    probe `GetMethod` rather than reason about it.
15. **No caching in an optimization unless the request explicitly allows it** — remove allocations and repeated
    enumeration first; propose a cache with its measurement and ask.
16. **Log through `Serilog.Modify.Log`** with structured templates (`"{Type}: … {CountyId}"`, never interpolation);
    `Log(exception, …)` always logs at Error.
17. **Rethrow only a requested cancellation:**
    `catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }` before any general catch.

### 2. Architecture (DiGi.Core pattern)
Data models are anemic; behavior is static extension methods. Never a service/manager class for behavior (process-wide
registries are `*Manager`s on `Settings` or DI singletons).
- **`/Classes`** (`[Project].Classes`), **`/Interfaces`**, **`/Enums`**. Category folders (`Classes/Options`,
  `/BackgroundTask`, `/Result`, `Convert/ToX`) add **no** namespace segment — it would change the `_type` discriminator.
- **`Query`** (read-only, property-like names: `BoundingBox()`, only `Is`/`Has`/`Try` prefixes), **`Modify`** (mutates),
  **`Create`** (builds), **`Convert`** (`/Convert/To[TargetArea]/[TargetType].cs`, `ToSystem_String`, `ToEPW_DateTime`) —
  `Convert` is the first choice for any conversion, never an instance method on the model.
- **Interface-contract members are instance methods on the class** — geometry primitives (`Polygon2D`, `Circle2D`, …
  implementing `IBoundable2D`, `ITransformable2D`, `IClosedCurve2D`) are deliberately not anemic; never move them.
- **Constructors stay cheap** (assign and clone only); validation and cleanup go in a `Create` factory named after the
  type, returning `null` for invalid input — materialise → clean up → then validate.
- **One public method per file** in `/Query`, `/Modify`, `/Create` (overloads together); nested types in
  `[Outer].[Inner].cs`; **no private methods** in these classes (public helper or local function).
- **`IResult` types end in `Result`** (`TerrainPointDensityResult`); the name is wire contract, so existing violations are
  not renamed opportunistically.
- Enums start with `Undefined` (`-1` only where `0` is real); tolerances come from `DiGi.Core.Constants.Tolerance`
  (`Distance` 1e-6, `MicroDistance`, `MacroDistance`, `Angle`).

### 3. Project assets — `files/` vs `user files/` (never commit secrets)
Committed, non-sensitive deployment assets go in `files/`; **secrets, credentials, machine-specific settings and
generated reports go in the git-ignored `user files/`** (`[Uu]ser [Ff]iles/` in `.gitignore`; verify with
`git check-ignore -v "user files/x.conf"`). Both are copied flat into the output (`CopyUserFiles` runs after `CopyFiles`,
so `user files/` wins) — and `user files/` **ships** with every deploy, so training data and model variants live outside
the workspace. A file a build copies into a project (`gltf-viewer-core.js`, `gis-theme.css`) is edited only at its source.

### 4. Host dependencies — `HintPath` drops a library's NuGet packages
DiGi projects reference each other by `HintPath`, which NuGet and MSBuild cannot follow: a host must **re-declare every
package its libraries need, at the same version** (unless its framework provides it), or it fails at run time with a
`FileNotFoundException` per item — a **partial result reported as success**. `dotnet test` never rebuilds the library
under test: build it first. Verify with `CheckHostDependencies.ps1`.

### 5. References (`IReference` / `IUniqueReference`)
**Never compare two interface-typed references with `==`** (a silent reference comparison; accessors clone per call).
Use `Core.Query.Equals(reference_1, reference_2)`.

### 6. Serialization (`SerializableObject`)
A serializable class implements `I<Project>SerializableObject`, keeps `private readonly` fields tagged
`[JsonInclude, JsonPropertyName(nameof(Prop))]`, exposes them as `[JsonIgnore]` get-only properties, uses
`DateTimeOffset` for dates, and has exactly three constructors:
```csharp
public Holiday(string? name, string? date)
{
    this.name = name;
    this.date = date;
}

public Holiday(Holiday? holiday) : base(holiday)
{
    if (holiday != null)
    {
        name = holiday.name;
        date = holiday.date;
    }
}

public Holiday(JsonObject? jsonObject) : base(jsonObject) { }
```
Copy constructor: primitive lists `new List<T>(source)`, nested objects `Core.Query.Clone(item)` element by element.

### 7. Documentation and tests
- **XML docs:** every public member, params in signature order, one `<summary>`, no blank lines (`<para>`), no docs on
  `partial` declarations. Read `documentation/API/<Assembly>/` before opening `.cs` source.
- **xUnit:** `[ProjectName].xUnit`, `public partial class Facts` in `/Facts`, a `<summary>` on every fact, shared fixtures
  in `DiGi.Test/files/`, written reports in `DiGi.Test/user files/reports/`. Every serializable class gets a fact ending in
  `Core.xUnit.Query.SerializationCheck(instance)` **plus** a direct copy-constructor assertion. Tolerance facts test just
  inside and just outside; benchmarks are measured in isolation with a warm-up. Slow facts are `[MediumFact]`/`[LongFact]`.
- **Interactive front ends** are verified in a real browser (Playwright), asserting rendered output, not state flags.

### 8. Branches and releases
Work is committed on the active version branch (the highest bare SemVer branch, e.g. `0.8.5`) — no feature branch,
worktree or pull request unless asked; push only when asked. **A version bump / release runs only on an explicit request:**
merge the version branch into `main`, bump the patch (`0.8.2` → `0.8.3`), branch it off `main`, update
`Directory.Build.props` `<Major>`/`<Minor>`/`<Build>`, push both (`git push -u origin <version_branch>`).

### 9. Full guideline set
Canonical, task-specific guidelines live in `DiGi.Maintenance/documentation/AI Guidelines/` (relative to the workspace
root) and are mirrored per repository as skills under `.agents/skills/<skill-name>/SKILL.md`.

| Guideline | Read it when |
|---|---|
| `Coding - General.md` | Writing or editing any C# code. |
| `Coding - Serialization.md` | Creating or changing a serializable type or options class; debugging a JSON round trip. |
| `Coding - Host Dependencies.md` | Adding a NuGet package; a host failing with `FileNotFoundException` or a partial result; `CS0012`/`CS0246` build walls. |
| `Coding - Project Setup.md` | New project or repository, `.editorconfig`, `Directory.Build.props`, analyzers. |
| `Coding - Automatic Tests.md` | Writing xUnit facts. |
| `Coding - Performance.md` | Optimizing, benchmarks, parity tests, the wiki `Benchmark.md` page. |
| `Coding - References.md` | Comparing or keying an `IReference` / `IUniqueReference`. |
| `Coding - Geometry.md` | Orienting, closing, triangulating, hashing or sweeping geometry. |
| `Coding - ComputeSharp.md` | Writing or changing a ComputeSharp `IComputeShader`. |
| `Coding - Grasshopper.md` | Grasshopper/Rhino plug-in components. |
| `Coding - Background Tasks.md` | Tray-application background tasks and their dialogs. |
| `Coding - WPF Deferred Loading.md` | A WPF window needing slow data; WPF layout checks. |
| `Coding - YOLO.md` | The YOLO Year Built detector. |
| `Coding - Laya.md` | The Laya decision model. |
| `Coding - WebAPI GLTF.md` | Web APIs on the `DiGi.GLTF` 3D framework. |
| `Coding - WebAPI Contracts.md` | Changing a WebAPI route/parameter/controller or writing an HTTP client. |
| `Coding - WebAPI Simple Authorization.md` | API-key tiered authorization. |
| `Coding - Deployed WebAPI.md` | Testing against `api.digiproject.uk`; which server runs what. |
| `Coding - Browser Testing.md` | Verifying interactive front-end behaviour in a real browser. |
| `Coding - GIS Administrative Data.md` | Anything keyed by a county code or id; the Subdivision layer. |
| `Coding - PostgreSQL.md` | PostgreSQL/Npgsql code, production vs development databases. |
| `Coding - PostgreSQL Distributed Queue Processing.md` | Distributed claim/acknowledge update queues. |
| `XML Documentation.md` | Reading generated API docs; adding or auditing XML docs. |
| `GitHub - Branches and Commits.md` | Where to commit, the pre-commit sweep, plan files, pulling all repos, releases. |
| `GitHub - Issues.md` | Creating, editing, verifying and closing issues; dependencies. |
| `GitHub - Labels.md` | Labels and the mandatory `ai: *` tier. |
| `GitHub - Sub-Issues.md` | Sub-issues for a multi-repository feature. |
| `GitHub - Workspace Setup.md` | Folder layout and new-machine setup. |
| `GitHub Wiki.md` | Editing wiki pages. |
| `Maintenance - Scripts.md` | `DiGi.Maintenance/Scripts`; scripted edits and line endings. |
| `Maintenance - AI Guidelines Propagation.md` | Adding, splitting or removing a guideline. |

> **One data rule worth knowing before it bites you:** a GIS county `code` does **not** identify one row — 406 county
> rows (one per polygon part) cover 380 codes. Key every write and lookup by county **`id`**, give every
> `LIMIT`/`FirstOrDefault` an `ORDER BY`, never deduplicate those rows, resolve every item's part against `building_2d`
> whatever number of ids arrived, and read building-keyed tables with `fallbackByReference: true`
> (`Coding - GIS Administrative Data.md`).
