import { FC } from 'react';
import ClipLoader from 'react-spinners/ClipLoader';

interface UploadProgressBarProps {
  progression: { loaded?: number; total?: number };
  cancel?: AbortController;
}

export const UploadProgressBar: FC<UploadProgressBarProps> = ({ progression, cancel }) => {
  return (
    <div id="progress-bar-window">
      <div id="progress-bar-container">
        <div className="horizontal center">
          <ClipLoader /> <span>Uploading dataset</span>{' '}
        </div>
        {progression.loaded && progression.total
          ? `${((progression.loaded / progression.total) * 100).toFixed(2)}%`
          : null}
        <progress id="upload-progress" value={progression.loaded} max={progression.total} />
        {cancel && (
          <button
            className="btn-submit-danger"
            onClick={() => {
              cancel.abort();
            }}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
