// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <trait_catalog.h>

#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

namespace
{
// ABI namespaces must not make the catalog depend on a particular library build.
[[nodiscard]] std::string qualified_name(const clang::NamedDecl& declaration)
{
  llvm::SmallVector<llvm::StringRef, 8> namespaces;
  for (const auto* context = declaration.getDeclContext(); !context->isTranslationUnit();
       context             = context->getParent())
  {
    const auto* ns = llvm::dyn_cast<clang::NamespaceDecl>(context);

    if (!ns || ns->isAnonymousNamespace())
    {
      return {};
    }

    if (!ns->isInline())
    {
      namespaces.push_back(ns->getName());
    }
  }
  if (namespaces.empty() || namespaces.back() != "cuda")
  {
    return {};
  }

  std::string result;
  for (auto it = namespaces.rbegin(); it != namespaces.rend(); ++it)
  {
    result += *it;
    result += "::";
  }
  result += declaration.getName();
  return result;
}

class trait_collector final : public clang::ast_matchers::MatchFinder::MatchCallback
{
public:
  llvm::StringSet<> bases;
  llvm::StringSet<> types;
  llvm::StringSet<> values;

  static constexpr llvm::StringRef DECL_BIND{""};

  void run(const clang::ast_matchers::MatchFinder::MatchResult& result) override
  {
    const auto* declaration = result.Nodes.getNodeAs<clang::NamedDecl>(DECL_BIND);
    if (const auto* variable = llvm::dyn_cast_or_null<clang::VarDecl>(declaration))
    {
      declaration = variable->getDescribedVarTemplate();
    }

    if (!declaration || !declaration->getDeclName().isIdentifier())
    {
      return;
    }

    const auto name = qualified_name(*declaration);

    if (name.empty())
    {
      return;
    }

    if (llvm::isa<clang::ClassTemplateDecl, clang::TypeAliasTemplateDecl>(declaration))
    {
      bases.insert(name);
    }

    if (llvm::isa<clang::TypeAliasTemplateDecl>(declaration) && declaration->getName().ends_with("_t"))
    {
      types.insert(llvm::StringRef{name}.drop_back(2));
    }

    if (llvm::isa<clang::VarTemplateDecl>(declaration) && declaration->getName().ends_with("_v"))
    {
      values.insert(llvm::StringRef{name}.drop_back(2));
    }
  }
};

void emit_set(llvm::raw_ostream& stream, llvm::StringRef name, const llvm::StringSet<>& traits)
{
  std::vector<llvm::StringRef> names;

  names.reserve(traits.size());
  for (const auto& entry : traits)
  {
    names.push_back(entry.getKey());
  }

  std::sort(names.begin(), names.end());

  stream << "// NOLINTNEXTLINE(bugprone-throwing-static-initialization)\n"
         << "const llvm::StringSet<> " << name << " = {\n";
  for (const auto entry : names)
  {
    stream << "  \"" << entry << "\",\n";
  }
  stream << "};\n\n";
}
} // namespace

int main(int argc, const char** argv)
{
  llvm::cl::OptionCategory options{"CCCL trait catalog"};
  const llvm::cl::opt<std::string> output{
    "output",
    llvm::cl::ValueOptional,
    llvm::cl::desc{
      "Write a merged catalog to <output> instead of checking for missing entries. A bare --output (or --output=-) "
      "defaults to stdout, while --output=<file> writes to <file> instead"},
    llvm::cl::cat{options}};

  auto parser = clang::tooling::CommonOptionsParser::create(argc, argv, options);
  if (!parser)
  {
    llvm::errs() << llvm::toString(parser.takeError()) << '\n';
    return 1;
  }

  trait_collector discovered;
  {
    using namespace clang::ast_matchers;

    MatchFinder finder;

    finder.addMatcher(
      namedDecl(anyOf(classTemplateDecl(), typeAliasTemplateDecl(), varDecl()), hasDeclContext(namespaceDecl()))
        .bind(trait_collector::DECL_BIND),
      &discovered);

    auto tool = clang::tooling::ClangTool{parser->getCompilations(), parser->getSourcePathList()};
    // Need to parse -resource-dir from the compile_commands output for the file, otherwise
    // clang tooling cannot find builtin headers like stdarg.h
    tool.appendArgumentsAdjuster([](const clang::tooling::CommandLineArguments& args, llvm::StringRef file) {
      return clang::tooling::getInsertArgumentAdjuster(
        ("-resource-dir=" + clang::GetResourcesPath(args.front())).c_str(),
        clang::tooling::ArgumentInsertPosition::BEGIN)(args, file);
    });

    auto fact = clang::tooling::newFrontendActionFactory(&finder);

    if (const auto status = tool.run(fact.get()); status != 0)
    {
      return status;
    }
  }

  // Keep entries from configurations that these translation units cannot expose.
  auto types   = clang::tidy::cccl::TYPE_TRAITS;
  auto values  = clang::tidy::cccl::VALUE_TRAITS;
  bool missing = false;

  const auto merge = [&](const llvm::StringSet<>& candidates, llvm::StringSet<>& catalog, llvm::StringRef name) {
    for (const auto& entry : candidates)
    {
      if (!discovered.bases.contains(entry.getKey()) || !catalog.insert(entry.getKey()).second)
      {
        continue;
      }

      missing = true;

      if (output.getNumOccurrences() == 0)
      {
        llvm::errs() << name << " is missing " << entry.getKey() << '\n';
      }
    }
  };

  merge(discovered.types, types, "TYPE_TRAITS");
  merge(discovered.values, values, "VALUE_TRAITS");

  for (llvm::StringRef req_type : {"cuda::std::remove_cvref", "cuda::std::remove_cv"})
  {
    if (!types.contains(req_type))
    {
      const auto msg =
        ("catalog does not contain " + req_type + ", this is a bug in the catalog generation program").str();

      llvm::reportFatalUsageError(msg.c_str());
    }
  }

  if (output.getNumOccurrences() == 0)
  {
    if (missing)
    {
      llvm::errs() << '\n';
      llvm::errs()
        << "Run to update inplace: " //
        << llvm::join(argv, argv + argc, " ")
        << " --output=" << llvm::sys::path::parent_path(__FILE__).str() + "/trait_catalog.h" << '\n';
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

  // "-" is parlance for stdout
  auto error = llvm::writeToOutput(output.empty() ? "-" : output.getValue(), [&](llvm::raw_ostream& stream) {
    stream << R"cpp(\
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated by generate_trait_catalog.cpp, do not edit.
//
// Note: entries absent from the scanned configurations are preserved because it is too hard to
// determine whether they actually disappeared or whether they are not hidden behind a
// configuration we can't see.

#pragma once

#include <llvm/ADT/StringSet.h>

namespace clang::tidy::cccl
{
)cpp";

    emit_set(stream, "TYPE_TRAITS", types);
    emit_set(stream, "VALUE_TRAITS", values);

    stream << "} // namespace clang::tidy::cccl\n";

    return llvm::Error::success();
  });

  if (error)
  {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
