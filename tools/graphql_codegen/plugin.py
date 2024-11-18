"""Defines a custom ariadne-codegen plugin to control Python code generation from GraphQL definitions.

For more info, see:
- https://github.com/mirumee/ariadne-codegen/blob/main/PLUGINS.md
- https://github.com/mirumee/ariadne-codegen/blob/main/ariadne_codegen/plugins/base.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import deque
from contextlib import suppress
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeGuard

from ariadne_codegen import Plugin
from graphlib import TopologicalSorter  # noqa # Run this only with python 3.9+
from graphql import GraphQLSchema

DEFAULT_BASE_MODEL_NAME = "BaseModel"  # The name of the default pydantic base class
CUSTOM_BASE_MODEL_NAME = "GQLBase"  # The name of our custom base class
TYPENAME = "Typename"  # The name of our custom Typename type for field annotations
ID = "ID"  # The name of the GraphQL ID type

#: Names that must be conditionally imported from `typing` or `typing_extensions` depending on python version.
TYPING_COMPAT_IMPORTS = frozenset({"override", "Annotated"})


class FixFragmentOrder(Plugin):
    """Codegen plugin to fix inconsistent ordering of fragments module.

    HACK: At the time of implementation, the fragments module has inconsistent ordering of
    - class definitions
    - `Class.model_rebuild()` statements

    See: https://github.com/mirumee/ariadne-codegen/issues/315. This plugin is a workaround in the meantime.
    """

    def generate_fragments_module(self, module: ast.Module, *_, **__) -> ast.Module:
        return self._ensure_class_order(module)

    @staticmethod
    def _ensure_class_order(module: ast.Module) -> ast.Module:
        # Separate the statements into the following expected groups:
        # - imports
        # - class definitions
        # - Model.model_rebuild() statements
        imports: deque[ast.Import | ast.ImportFrom] = deque()
        class_defs: deque[ast.ClassDef] = deque()
        model_rebuilds: deque[ast.Expr] = deque()

        for stmt in module.body:
            if is_import(stmt) or is_import_from(stmt):
                imports.append(stmt)
            elif is_class_def(stmt):
                class_defs.append(stmt)
            elif is_model_rebuild(stmt):
                model_rebuilds.append(stmt)
            else:
                stmt_repr = ast.unparse(stmt)
                raise TypeError(f"Unexpected {type(stmt)!r} statement:\n{stmt_repr}")

        # Deterministically reorder the class definitions, ensuring parent classes are defined first
        sorter = ClassDefSorter(class_defs)
        class_defs = sorter.sort_class_defs(class_defs)
        model_rebuilds = sorter.sort_model_rebuilds(model_rebuilds)

        module.body = [*imports, *class_defs, *model_rebuilds]
        return module


class ClassDefSorter:
    """A sorter for a collection of class definitions."""

    toposorter: TopologicalSorter
    static_order: tuple[str, ...]
    name2idx: dict[str, int]

    def __init__(self, class_defs: Iterable[ast.ClassDef]) -> None:
        self.toposorter = TopologicalSorter()

        # Sort the class definitions by name first to ensure deterministic ordering
        for class_def in sorted(class_defs, key=lambda cls: cls.name):
            class_name = class_def.name
            base_names = [base.id for base in class_def.bases]
            self.toposorter.add(class_name, *base_names)

        self.static_order = tuple(self.toposorter.static_order())
        self.name2idx = {name: idx for idx, name in enumerate(self.static_order)}

    def sort_class_defs(self, class_defs: Iterable[ast.ClassDef]) -> list[ast.ClassDef]:
        """Return the class definitions in topologically sorted order."""
        return sorted(class_defs, key=lambda stmt: self.name2idx[stmt.name])

    def sort_model_rebuilds(self, model_rebuilds: Iterable[ast.Expr]) -> list[ast.Expr]:
        """Return the model rebuild statements in topologically sorted order."""
        return sorted(
            model_rebuilds, key=lambda stmt: self.name2idx[stmt.value.func.value.id]
        )


def forget_default_id_type() -> None:
    # HACK: Override the default python type that ariadne-codegen uses for GraphQL's `ID` type.
    # See: https://github.com/mirumee/ariadne-codegen/issues/316
    from ariadne_codegen.client_generators import constants as codegen_constants

    with suppress(LookupError):
        codegen_constants.SIMPLE_TYPE_MAP.pop(ID)
    with suppress(LookupError):
        codegen_constants.INPUT_SCALARS_MAP.pop(ID)


class GraphQLCodegenPlugin(Plugin):
    """An `ariadne-codegen` plugin to customize generated Python code for the `wandb` package.

    For more info about allowed methods, see:
    - https://github.com/mirumee/ariadne-codegen/blob/main/PLUGINS.md
    - https://github.com/mirumee/ariadne-codegen/blob/main/ariadne_codegen/plugins/base.py
    """

    # Inherited
    schema: GraphQLSchema
    config_dict: dict[str, Any]

    #: The directory where the generated modules will be added
    package_dir: Path
    #: Generated classes that we don't need in the final code
    classes_to_drop: set[str]
    #: Generated modules that we don't need in the final code
    modules_to_drop: frozenset[str]

    #: A NodeTransformer to replace `pydantic.BaseModel` with `GQLBase`
    _base_model_replacer: BaseModelReplacer

    def __init__(self, schema: GraphQLSchema, config_dict: dict[str, Any]) -> None:
        super().__init__(schema, config_dict)

        codegen_config: dict[str, Any] = config_dict["tool"]["ariadne-codegen"]

        package_path = codegen_config["target_package_path"]
        package_name = codegen_config["target_package_name"]
        self.package_dir = Path(package_path) / package_name

        self.classes_to_drop = set()
        self.modules_to_drop = frozenset(codegen_config["modules_to_drop"])

        # HACK: Override the default python type that ariadne-codegen uses for GraphQL's `ID` type.
        # See: https://github.com/mirumee/ariadne-codegen/issues/316
        if ID in codegen_config["scalars"]:
            forget_default_id_type()

        self._base_model_replacer = BaseModelReplacer()

    def generate_init_code(self, generated_code: str) -> str:
        # This should be the last hook in the codegen process, after all modules have been generated.
        # So at this step, perform cleanup like ...
        self._remove_module_files(self.modules_to_drop)  # Omit modules
        self._apply_ruff(self.package_dir)  # Apply auto-formatting

        return super().generate_init_code(generated_code)

    def _remove_module_files(self, modules_to_drop: Iterable[str]) -> None:
        sys.stdout.write("\n========== Removing files we don't need ==========\n")
        for module_name in modules_to_drop:
            module_path = (self.package_dir / module_name).with_suffix(".py")
            with suppress(FileNotFoundError):
                module_path.unlink()
                sys.stdout.write(f"Removed unused module: {module_path!s}\n")

    @staticmethod
    def _apply_ruff(path: str | Path) -> None:
        path = str(path)
        sys.stdout.write(f"\n========== Reformatting: {path} ==========\n")
        subprocess.run(["ruff", "check", "--fix", "--unsafe-fixes", path], check=True)
        subprocess.run(["ruff", "format", path], check=True)

    @staticmethod
    def _add_common_imports(module: ast.Module) -> ast.Module:
        """Return a copy of the parse module after inserting common import statements."""
        module.body = [
            # `from __future__ import annotations`
            # `import sys`
            # `from .base import GQLBase, Typename, etc.`
            make_import_from("__future__", "annotations"),
            make_import("sys"),
            make_import_from("base", [CUSTOM_BASE_MODEL_NAME, TYPENAME], level=1),
            *module.body,
        ]
        return module

    def generate_init_module(self, module: ast.Module) -> ast.Module:
        return self._cleanup_init_module(module)

    def generate_enums_module(self, module: ast.Module) -> ast.Module:
        module = self._add_common_imports(module)
        module = self._base_model_replacer.visit(module)
        module = self._replace_redundant_classes(module)
        return ast.fix_missing_locations(module)

    def generate_inputs_module(self, module: ast.Module) -> ast.Module:
        module = self._add_common_imports(module)
        module = self._base_model_replacer.visit(module)
        module = self._replace_redundant_classes(module)
        return ast.fix_missing_locations(module)

    def generate_result_types_module(self, module: ast.Module, *_, **__) -> ast.Module:
        module = self._add_common_imports(module)
        module = self._base_model_replacer.visit(module)
        module = self._replace_redundant_classes(module)
        return ast.fix_missing_locations(module)

    def generate_fragments_module(self, module: ast.Module, *_, **__) -> ast.Module:
        module = self._add_common_imports(module)
        module = self._base_model_replacer.visit(module)
        module = self._replace_redundant_classes(module)
        return ast.fix_missing_locations(module)

    def _replace_redundant_classes(self, module: ast.Module) -> ast.Module:
        # Identify redundant classes and build replacement mapping
        redundant_class_defs = filter(is_redundant_subclass_def, module.body)

        class_name_replacements = {
            # maps names of: redundant subclass -> parent class
            class_def.name: class_def.bases[0].id
            for class_def in redundant_class_defs
        }

        # Record removed classes for later cleanup
        self.classes_to_drop.update(class_name_replacements.keys())

        # Update any references to redundant classes in the remaining class definitions
        # Replace the module body with the cleaned-up statements
        return RedundantClassReplacer(class_name_replacements).visit(module)

    def _cleanup_init_module(self, module: ast.Module) -> ast.Module:
        """Clean up the __init__ module by removing dropped imports and rewriting the `__all__` exports."""
        # Drop selected import statements from the __init__ module
        orig_stmts = module.body
        kept_stmts = list(self._filter_init_imports(orig_stmts))

        # Rewrite the `__all__` exports
        names_to_export = self._collect_init_exports(kept_stmts)
        export_stmt = self._generate_init_export(names_to_export)

        # Update the module body with the cleaned-up statements
        module.body = [*kept_stmts, export_stmt]
        return ast.fix_missing_locations(module)

    def _generate_init_export(self, names: Iterable[str]) -> ast.Assign:
        """Generate an `__all__ = [...]` statement to export the given names from __init__.py."""
        return make_assign(
            "__all__",
            ast.List([ast.Constant(name) for name in names]),
        )

    def _collect_init_exports(self, stmts: Iterable[ast.stmt]) -> list[str]:
        """Return set of names to export from the __init__ module."""
        return list(
            chain.from_iterable(
                sorted(imported_names(import_from))
                for import_from in filter(is_import_from, stmts)
            )
        )

    def _filter_init_imports(self, stmts: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
        """Yield only import statements to keep in the __init__ module."""
        import_from_stmts = filter(is_import_from, stmts)
        for stmt in import_from_stmts:
            if stmt.module not in self.modules_to_drop:
                # Keep only imported names that aren't being dropped
                kept_names = sorted(imported_names(stmt) - self.classes_to_drop)
                yield make_import_from(stmt.module, kept_names, level=1)


class FixImports(Plugin):
    def generate_enums_module(self, module: ast.Module) -> ast.Module:
        return self._fix_typing_imports(module)

    def generate_inputs_module(self, module: ast.Module) -> ast.Module:
        return self._fix_typing_imports(module)

    def generate_result_types_module(self, module: ast.Module, *_, **__) -> ast.Module:
        return self._fix_typing_imports(module)

    def generate_fragments_module(self, module: ast.Module, *_, **__) -> ast.Module:
        return self._fix_typing_imports(module)

    @staticmethod
    def _fix_typing_imports(module: ast.Module) -> ast.Module:
        import_stmts: deque[ast.stmt] = deque()
        other_stmts: deque[ast.stmt] = deque()

        type_reimports: set[str] | None = None

        for stmt in module.body:
            # Handle `from typing import ...` statements
            if is_import_from(stmt) and (stmt.module == "typing"):
                type_imports = imported_names(stmt)

                if kept_imports := (type_imports - TYPING_COMPAT_IMPORTS):
                    new_stmt = make_import_from("typing", kept_imports)
                    import_stmts.append(new_stmt)

                # Insert the typing_compat reimports, if needed
                if type_reimports := (type_imports & TYPING_COMPAT_IMPORTS):
                    compat_import_stmt = make_import_from(
                        "typing_compat", type_reimports, level=1
                    )
                    import_stmts.append(compat_import_stmt)

            # Keep all other import statements
            elif is_import(stmt) or is_import_from(stmt):
                import_stmts.append(stmt)

            # Keep all other statements
            else:
                other_stmts.append(stmt)

        module.body = [*import_stmts, *other_stmts]
        return module


class BaseModelReplacer(ast.NodeTransformer):
    """Replaces all `pydantic.BaseModel` base classes with `GQLBase`."""

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        # Don't rewrite imports of the BaseModel class, those'll get dropped later
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit the name of a base class in a class definition."""
        if node.id == DEFAULT_BASE_MODEL_NAME:
            node.id = CUSTOM_BASE_MODEL_NAME
        return self.generic_visit(node)


class RedundantClassReplacer(ast.NodeTransformer):
    """Removes redundant class definitions and references to them."""

    #: Maps deleted class names -> replacement class names
    replacement_names: dict[str, str]

    def __init__(self, replacement_names: dict[str, str]):
        self.replacement_names = replacement_names

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.name in self.replacement_names:
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.target.id == "typename__":
            # Rewrite e.g.
            # - before: `typename__: Literal["MyType"] = Field(...)`
            # - after:  `typename__: Typename[Literal["MyType"]]`
            node.annotation = ast.Subscript(ast.Name(id=TYPENAME), node.annotation)
            node.value = None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        # node.id may be the name of the hinted type, e.g. `MyType`
        # or an implicit forward ref, e.g. `"MyType"`, `'MyType'`
        unquoted_name = node.id.strip("'\"")
        with suppress(LookupError):
            node.id = self.replacement_names[unquoted_name]
        return self.generic_visit(node)


# Custom helpers
def imported_names(stmt: ast.Import | ast.ImportFrom) -> set[str]:
    """Return the (str) names imported by this `from ... import {names}` statement."""
    return {alias.name for alias in stmt.names}


def is_redundant_subclass_def(stmt: ast.ClassDef) -> TypeGuard[ast.ClassDef]:
    """Return True if this class definition is a redundant subclass definition.

    A redundant subclass will look like:
        class MyClass(ParentClass):
            pass

    is redundant if it has only one base class, and
    """
    return (
        is_class_def(stmt)
        and isinstance(stmt.body[0], ast.Pass)
        and len(stmt.bases) == 1
    )


def is_all_assignment(stmt: ast.stmt) -> TypeGuard[ast.Assign]:
    """Return True if this node is an assignment statement to `__all__ = [...]`."""
    return (
        isinstance(stmt, ast.Assign)
        and (stmt.targets[0].id == "__all__")
        and isinstance(stmt.value, ast.List)
    )


def is_class_def(stmt: ast.stmt) -> TypeGuard[ast.ClassDef]:
    """Return True if this node is a class definition."""
    return isinstance(stmt, ast.ClassDef)


def is_import(stmt: ast.stmt) -> TypeGuard[ast.Import]:
    """Return True if this node is an `import ...` statement."""
    return isinstance(stmt, ast.Import)


def is_import_from(stmt: ast.stmt) -> TypeGuard[ast.ImportFrom]:
    """Return True if this node is a `from ... import ...` statement."""
    return isinstance(stmt, ast.ImportFrom)


def is_model_rebuild(node: ast.stmt) -> TypeGuard[ast.Expr]:
    """Return True if this node is a generated `PydanticModel.model_rebuild()` statement.

    A module-level statement like:
        MyModel.model_rebuild()

    will be an AST node like:
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(id='MyModel'),
                    attr='model_rebuild',
                ), ...
            ),
        )
    """
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and (node.value.func.attr == "model_rebuild")
    )


def make_assign(target: str, value: ast.expr) -> ast.Assign:
    """Generate the AST node for an `{target} = {value}` assignment statement."""
    return ast.Assign(targets=[ast.Name(id=target)], value=value)


def make_import(modules: str | Iterable[str]) -> ast.Import:
    """Generate the AST node for an `import {modules}` statement."""
    modules = [modules] if isinstance(modules, str) else modules
    return ast.Import(names=[ast.alias(name) for name in modules])


def make_import_from(
    module: str, names: str | Iterable[str], level: int = 0
) -> ast.ImportFrom:
    """Generate the AST node for a `from {module} import {names}` statement."""
    names = [names] if isinstance(names, str) else names
    return ast.ImportFrom(
        module=module, names=[ast.alias(name) for name in names], level=level
    )
