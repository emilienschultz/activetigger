import { FC, useCallback, useEffect, useState } from 'react';
import { FaCheck } from 'react-icons/fa6';
import Select from 'react-select';
import { ElementOutModel } from '../../types';

interface MultilabelInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string, comment?: string) => void;
  element?: ElementOutModel;
}

export const MultilabelInput: FC<MultilabelInputProps> = ({
  elementId,
  postAnnotation,
  labels,
  element,
}) => {
  // management multilabels
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [comment, setComment] = useState<string>(
    element?.history ? element.history[0]?.comment || '' : '',
  );

  useEffect(
    () => setComment(element?.history ? element.history[0]?.comment || '' : ''),
    [element],
  );

  // add shortcut
  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (ev.key === 'Enter' && ev.ctrlKey) {
        postAnnotation(selectedLabels.join('|'), elementId, comment);
        setSelectedLabels([]);
      }
    },
    [postAnnotation, selectedLabels, elementId, comment],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyboardEvents);
    return () => {
      document.removeEventListener('keydown', handleKeyboardEvents);
    };
  }, [handleKeyboardEvents]);

  return (
    <div className="d-flex flex-column gap-2 w-100">
      <div className="d-flex gap-2 align-items-center w-100">
        <Select
          isMulti
          options={labels.map((e) => ({ value: e, label: e }))}
          onChange={(e) => {
            setSelectedLabels(e.map((e) => e.value));
          }}
          value={selectedLabels.map((e) => ({ value: e, label: e }))}
          className="w-100"
        />
        <button
          className="btn btn-outline-success d-flex align-items-center justify-content-center validate-btn"
          onClick={() => {
            postAnnotation(selectedLabels.join('|'), elementId, comment);
            setSelectedLabels([]);
          }}
        >
          <FaCheck size={18} />
        </button>
      </div>
      <textarea
        className="form-control annotation-comment"
        placeholder="Comment"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
    </div>
  );
};
