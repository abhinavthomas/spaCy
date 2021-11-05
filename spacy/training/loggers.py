import re
from typing import TYPE_CHECKING, Dict, Any, Tuple, Callable, List, Optional, IO
from wasabi import Printer
import tqdm
import sys

from ..util import registry
from .. import util
from ..errors import Errors

if TYPE_CHECKING:
    from ..language import Language  # noqa: F401


def setup_table(
    *, cols: List[str], widths: List[int], max_width: int = 13
) -> Tuple[List[str], List[int], List[str]]:
    final_cols = []
    final_widths = []
    for col, width in zip(cols, widths):
        if len(col) > max_width:
            col = col[: max_width - 3] + "..."  # shorten column if too long
        final_cols.append(col.upper())
        final_widths.append(max(len(col), width))
    return final_cols, final_widths, ["r" for _ in final_widths]


@registry.loggers("spacy.ConsoleLogger.v1")
def console_logger(progress_bar: bool = False):
    def setup_printer(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Optional[Dict[str, Any]]], None], Callable[[], None]]:
        def write(text): return print(text, file=stdout, flush=True)
        msg = Printer(no_print=True)
        # ensure that only trainable components are logged
        logged_pipes = [
            name
            for name, proc in nlp.pipeline
            if hasattr(proc, "is_trainable") and proc.is_trainable
        ]
        eval_frequency = nlp.config["training"]["eval_frequency"]
        score_weights = nlp.config["training"]["score_weights"]
        score_cols = [col for col, value in score_weights.items()
                      if value is not None]
        loss_cols = [f"Loss {pipe}" for pipe in logged_pipes]
        spacing = 2
        table_header, table_widths, table_aligns = setup_table(
            cols=["E", "#"] + loss_cols + score_cols + ["Score"],
            widths=[3, 6] + [8 for _ in loss_cols]
            + [6 for _ in score_cols] + [6],
        )
        write(msg.row(table_header, widths=table_widths, spacing=spacing))
        write(msg.row(["-" * width for width in table_widths], spacing=spacing))
        progress = None

        def log_step(info: Optional[Dict[str, Any]]) -> None:
            nonlocal progress

            if info is None:
                # If we don't have a new checkpoint, just return.
                if progress is not None:
                    progress.update(1)
                return
            losses = [
                "{0:.2f}".format(float(info["losses"][pipe_name]))
                for pipe_name in logged_pipes
            ]

            scores = []
            for col in score_cols:
                score = info["other_scores"].get(col, 0.0)
                try:
                    score = float(score)
                except TypeError:
                    err = Errors.E916.format(name=col, score_type=type(score))
                    raise ValueError(err) from None
                if col != "speed":
                    score *= 100
                scores.append("{0:.2f}".format(score))

            data = (
                [info["epoch"], info["step"]]
                + losses
                + scores
                + ["{0:.2f}".format(float(info["score"]))]
            )
            if progress is not None:
                progress.close()
            write(
                msg.row(data, widths=table_widths,
                        aligns=table_aligns, spacing=spacing)
            )
            if progress_bar:
                # Set disable=None, so that it disables on non-TTY
                progress = tqdm.tqdm(
                    total=eval_frequency, disable=None, leave=False, file=stderr
                )
                progress.set_description(f"Epoch {info['epoch']+1}")

        def finalize() -> None:
            pass

        return log_step, finalize

    return setup_printer


@registry.loggers("spacy.WandbLogger.v2")
def wandb_logger_v2(
    project_name: str,
    remove_config_values: List[str] = [],
    model_log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
):
    try:
        import wandb

        # test that these are available
        from wandb import init, log, join  # noqa: F401
    except ImportError:
        raise ImportError(Errors.E880)

    console = console_logger(progress_bar=False)

    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        config = nlp.config.interpolate()
        config_dot = util.dict_to_dot(config)
        for field in remove_config_values:
            del config_dot[field]
        config = util.dot_to_dict(config_dot)
        run = wandb.init(project=project_name, config=config, reinit=True)
        console_log_step, console_finalize = console(nlp, stdout, stderr)

        def log_dir_artifact(
            path: str,
            name: str,
            type: str,
            metadata: Optional[Dict[str, Any]] = {},
            aliases: Optional[List[str]] = [],
        ):
            dataset_artifact = wandb.Artifact(
                name, type=type, metadata=metadata)
            dataset_artifact.add_dir(path, name=name)
            wandb.log_artifact(dataset_artifact, aliases=aliases)

        if log_dataset_dir:
            log_dir_artifact(path=log_dataset_dir,
                             name="dataset", type="dataset")

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info is not None:
                score = info["score"]
                other_scores = info["other_scores"]
                losses = info["losses"]
                wandb.log({"score": score})
                if losses:
                    wandb.log({f"loss_{k}": v for k, v in losses.items()})
                if isinstance(other_scores, dict):
                    wandb.log(other_scores)
                if model_log_interval and info.get("output_path"):
                    if info["step"] % model_log_interval == 0 and info["step"] != 0:
                        log_dir_artifact(
                            path=info["output_path"],
                            name="pipeline_" + run.id,
                            type="checkpoint",
                            metadata=info,
                            aliases=[
                                f"epoch {info['epoch']} step {info['step']}",
                                "latest",
                                "best"
                                if info["score"] == max(info["checkpoints"])[0]
                                else "",
                            ],
                        )

        def finalize() -> None:
            console_finalize()
            wandb.join()

        return log_step, finalize

    return setup_logger


@registry.loggers("spacy.WandbLogger.v3")
def wandb_logger_v3(
    project_name: str,
    remove_config_values: List[str] = [],
    model_log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
):
    try:
        import wandb

        # test that these are available
        from wandb import init, log, join  # noqa: F401
    except ImportError:
        raise ImportError(Errors.E880)

    console = console_logger(progress_bar=False)

    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        config = nlp.config.interpolate()
        config_dot = util.dict_to_dot(config)
        for field in remove_config_values:
            del config_dot[field]
        config = util.dot_to_dict(config_dot)
        run = wandb.init(
            project=project_name, config=config, entity=entity, reinit=True
        )

        if run_name:
            wandb.run.name = run_name

        console_log_step, console_finalize = console(nlp, stdout, stderr)

        def log_dir_artifact(
            path: str,
            name: str,
            type: str,
            metadata: Optional[Dict[str, Any]] = {},
            aliases: Optional[List[str]] = [],
        ):
            dataset_artifact = wandb.Artifact(
                name, type=type, metadata=metadata)
            dataset_artifact.add_dir(path, name=name)
            wandb.log_artifact(dataset_artifact, aliases=aliases)

        if log_dataset_dir:
            log_dir_artifact(path=log_dataset_dir,
                             name="dataset", type="dataset")

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info is not None:
                score = info["score"]
                other_scores = info["other_scores"]
                losses = info["losses"]
                wandb.log({"score": score})
                if losses:
                    wandb.log({f"loss_{k}": v for k, v in losses.items()})
                if isinstance(other_scores, dict):
                    wandb.log(other_scores)
                if model_log_interval and info.get("output_path"):
                    if info["step"] % model_log_interval == 0 and info["step"] != 0:
                        log_dir_artifact(
                            path=info["output_path"],
                            name="pipeline_" + run.id,
                            type="checkpoint",
                            metadata=info,
                            aliases=[
                                f"epoch {info['epoch']} step {info['step']}",
                                "latest",
                                "best"
                                if info["score"] == max(info["checkpoints"])[0]
                                else "",
                            ],
                        )

        def finalize() -> None:
            console_finalize()
            wandb.join()

        return log_step, finalize

    return setup_logger


@registry.loggers("spacy.MLFlow.v1")
def mlflow_logger_v1(
    remove_config_values: List[str] = [],
    model_log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    log_best_model: Optional[bool] = False
):
    try:
        import mlflow
    except ImportError:
        raise ImportError(Errors.E880)

    console = console_logger(progress_bar=False)

    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        config = nlp.config.interpolate()
        config_dot = util.dict_to_dot(config)
        for field in remove_config_values:
            del config_dot[field]
        config = util.dot_to_dict(config_dot)
        _ml_flow = mlflow

        _ml_flow.start_run(run_name=run_name,
                           nested=_ml_flow.active_run() is not None)

        # Required as MLFlow doesn't support @ to be part of keys of parameters and metrics
        pattern = re.compile('[\W]+')

        def log_params(param: Dict[str, Any], parent: str = ''):
            for k, v in param.items():
                if isinstance(v, (dict,)):
                    if parent:
                        log_params(v, f"{parent}.{k}")
                    else:
                        log_params(v, k)
                else:
                    if parent:
                        _ml_flow.log_param(
                            pattern.sub('.', f"{parent}.{k}"), v)
                    else:
                        _ml_flow.log_param(pattern.sub('.', k), v)

        log_params(config)

        def log_metric(metric: Dict[str, Any], parent: str = '', step: Optional[int] = None):
            for k, v in metric.items():
                if isinstance(v, (dict,)):
                    if parent:
                        log_metric(v, f"{parent}.{k}", step)
                    else:
                        log_metric(v, k, step)
                else:
                    if parent:
                        _ml_flow.log_metric(pattern.sub(
                            '.', f"{parent}.{k}"), v, step=step)
                    else:
                        _ml_flow.log_metric(pattern.sub('.', k), v, step=step)

        console_log_step, console_finalize = console(nlp, stdout, stderr)

        if log_dataset_dir:
            _ml_flow.log_artifact(log_dataset_dir, "dataset")

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info is not None:
                if model_log_interval and info.get("output_path"):
                    if info["step"] % model_log_interval == 0 and info["step"] != 0:
                        metrics = {k: info.get(k) for k in (
                            "losses", "score", "other_scores")}
                        log_metric(metrics, step=info.get("step"))
                        _ml_flow.spacy.log_model(
                            nlp, f"models/checkpoints/epoch_{info['epoch']}_step_{info['step']}")
                        if log_best_model and info["score"] == max(info["checkpoints"])[0]:
                            _ml_flow.spacy.log_model(
                                nlp, f"models/best-model")

        def finalize() -> None:
            console_finalize()
            _ml_flow.end_run()

        return log_step, finalize

    return setup_logger
