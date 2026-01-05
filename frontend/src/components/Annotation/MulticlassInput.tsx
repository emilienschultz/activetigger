import cx from 'classnames';
import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { IoIosRadioButtonOff, IoIosRadioButtonOn } from 'react-icons/io';
import { MdOnlinePrediction } from 'react-icons/md';
import { Tooltip } from 'react-tooltip';
import { useAppContext } from '../../core/context';
import { reorderLabels } from '../../core/utils';
import { ElementOutModel } from '../../types';
import { NoAnnotationIcon } from '../Icons';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string | null, elementId: string, comment?: string) => void;
  fetchToNextAnnotation?: (elementId: string) => void;
  small?: boolean;
  phase?: string;
  element?: ElementOutModel;
}

interface LabelType {
  id: number;
  label: string;
}

export const MulticlassInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  fetchToNextAnnotation,
  labels,
  phase,
  element,
  small = false,
}) => {
  // get the context and set the labels
  const {
    appContext: { displayConfig, activeModel },
  } = useAppContext();

  //TODO: grab comment from element history once API has been modified to add this
  const [comment, setComment] = useState<string>('');

  //reset comment as for now it's not available in annotation history
  useEffect(() => setComment(''), [elementId]);

  const availableLabels = useMemo<LabelType[]>(
    () =>
      reorderLabels(labels || [], displayConfig.labelsOrder || []).map((label, index) => ({
        id: index,
        label: label,
      })),
    [displayConfig.labelsOrder, labels],
  );

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      console.log(ev.code);
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (activeModel && ev.code === 'KeyP') {
        postAnnotation(element?.predict.label || '', elementId, comment);
      }
      if (ev.code === 'KeyS' && fetchToNextAnnotation) {
        fetchToNextAnnotation(elementId);
      }
      if (ev.code === 'Delete') {
        postAnnotation(null, elementId, comment);
      }

      availableLabels.forEach((item, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          postAnnotation(item.label, elementId, comment);
        }
      });
    },
    [
      availableLabels,
      postAnnotation,
      elementId,
      element,
      activeModel,
      fetchToNextAnnotation,
      comment,
    ],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    if (availableLabels.length > 0 && availableLabels.length < 10) {
      document.addEventListener('keydown', handleKeyboardEvents);
    }

    return () => {
      if (availableLabels.length > 0 && availableLabels.length < 10) {
        document.removeEventListener('keydown', handleKeyboardEvents);
      }
    };
  }, [availableLabels, handleKeyboardEvents]);

  const predict_proba = element?.predict.proba ? element.predict.proba.toFixed(2) : 'NA';
  const predict_entropy = element?.predict.entropy ? element.predict.entropy.toFixed(2) : 'NA';

  const lastAnnotation = useMemo(() => {
    return element?.history?.length && element?.history.length > 0
      ? (element?.history[0] as string[])
      : null;
  }, [element?.history]);

  return (
    <div className="d-flex flex-column justify-content-center justify-content-lg-start gap-3">
      {/* TAGS ACTIONS */}
      <div className="d-flex flex-row flex-lg-column justify-content-center justify-content-lg-start flex-wrap gap-2 align-items-end align-items-lg-start">
        {
          // display buttons for label from the user
          availableLabels.map((e, i) => (
            <button
              type="button"
              key={e.label}
              value={e.label}
              className="btn-annotate-action"
              onClick={(v) => {
                postAnnotation(v.currentTarget.value, elementId, comment);
              }}
            >
              {displayConfig.displayAnnotation ? (
                lastAnnotation && lastAnnotation[0] === e.label ? (
                  <IoIosRadioButtonOn />
                ) : (
                  <IoIosRadioButtonOff />
                )
              ) : null}
              <span className="text-truncate" style={{ maxWidth: '20vw' }}>
                {e.label}
              </span>{' '}
              <span className="badge hotkey">{i + 1}</span>
            </button>
          ))
        }
        {/* NO TAG OPTION */}
        <button
          type="button"
          className="btn-annotate-action"
          onClick={() => {
            postAnnotation(null, elementId, comment);
          }}
        >
          {lastAnnotation === null ? <IoIosRadioButtonOn /> : <IoIosRadioButtonOff />}
          <NoAnnotationIcon /> No tag <span className="badge hotkey">DEL</span>
        </button>
        {/* PREDICTION */}
        {phase == 'train' && displayConfig.displayPrediction && element?.predict.label && (
          <div className="d-flex flex-column align-items-start gap-1">
            <small className="d-flex align-items-center gap-1">
              <MdOnlinePrediction size="20" title="Prediction by model" id="prediction-icon" />{' '}
              <Tooltip anchorSelect="#prediction-icon" place="top">
                Prediction by model
              </Tooltip>
              <span className="badge m-0" id="predict-probability">
                P. {predict_proba}
              </span>
              <Tooltip anchorSelect="#predict-probability" place="top">
                prediction's probability: {predict_proba}
              </Tooltip>
              <span className="badge m-0" id="predict-entropy">
                E. {predict_entropy}
              </span>
              <Tooltip anchorSelect="#predict-entropy" place="top">
                prediction's entropy: {predict_entropy}
              </Tooltip>
            </small>
            <button
              type="button"
              value={element?.predict.label as unknown as string}
              className={cx('btn-annotate-predicted-action', small ? ' icon-small' : '')}
              onClick={(e) => {
                postAnnotation(e.currentTarget.value, elementId);
              }}
            >
              {element?.predict.label} <span className="badge hotkey">P</span>
            </button>
          </div>
        )}
      </div>

      <div>
        <textarea
          className="form-control"
          placeholder="Comment"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
      </div>
      {/* EXTRA ACTIONS */}
      <div className="d-flex flex-row flex-lg-column justify-content-center justify-content-lg-start flex-wrap flex-lg-nowrap">
        {/* SKIP */}
        {fetchToNextAnnotation && (
          <button
            type="button"
            className="btn-annotate-action h-100"
            onClick={() => {
              fetchToNextAnnotation(elementId);
            }}
          >
            Skip <span className="badge hotkey">S</span>
          </button>
        )}
      </div>
    </div>
  );
};
