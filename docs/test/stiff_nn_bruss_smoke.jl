using Documenter

const DOCS_ROOT = normpath(joinpath(@__DIR__, ".."))
const PAGE = joinpath(DOCS_ROOT, "src", "examples", "stiff_nn_bruss.md")
const NAVIGATION = read(joinpath(DOCS_ROOT, "pages.jl"), String)
const PAGE_TEXT = read(PAGE, String)

@assert occursin("\"examples/stiff_nn_bruss.md\"", NAVIGATION)
@assert count("```@example stiff_bruss", PAGE_TEXT) == 6

mktempdir() do root
    mkpath(joinpath(root, "src"))
    write(joinpath(root, "src", "index.md"), "# Tutorial smoke test\n")
    cp(PAGE, joinpath(root, "src", "stiff_nn_bruss.md"))
    ENV["GKSwstype"] = "100"
    started = time()
    makedocs(
        root = root,
        source = "src",
        build = "build",
        sitename = "DiffEqFlux tutorial smoke test",
        modules = Module[],
        pages = ["Stiff Brusselator" => "stiff_nn_bruss.md"],
        format = Documenter.HTML(repolink = "https://github.com/SciML/DiffEqFlux.jl"),
        remotes = nothing,
        version = "smoke",
    )
    elapsed = time() - started
    built_page = any(
        "stiff_nn_bruss.html" in files ||
            (basename(dir) == "stiff_nn_bruss" && "index.html" in files)
            for (dir, _, files) in walkdir(joinpath(root, "build"))
    )
    @assert built_page
    println("Docs smoke completed successfully in $(round(elapsed; digits = 2)) seconds")
end
