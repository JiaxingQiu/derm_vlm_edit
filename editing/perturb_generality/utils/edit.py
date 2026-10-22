"""Build a ReasonEdit codebook from an edit list. Keys use the original image and question."""

from __future__ import annotations


def edit(vlm, config, edits):
    from derm_dataset import DermEditDataset
    from reasonedit import ReasonEdit

    queries = DermEditDataset(config, edits)
    queries.set_dataloader()
    editor = ReasonEdit(config, vlm)
    editor.edit(config, edit_ds=queries)
    return editor, queries
