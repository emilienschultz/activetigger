import { FC } from 'react';
import ClipLoader from 'react-spinners/ClipLoader';

interface UploadProgressBarProps {
  progression: { loaded?: number; total?: number };
  cancel?: AbortController;
}

export const UploadProgressBar: FC<UploadProgressBarProps> = ({ progression, cancel }) => {
  return (
    <div>
      <div className="position-absolute bg-white w-100 h-100 top-0 left-0 d-flex flex-column justify-content-center bg-opacity-50">
        <div className="d-flex flex-column bg-white p-4 border border-dark gap-2">
          <div className="d-flex align-items-center gap-2 ">
            <ClipLoader /> <span>Uploading dataset</span>{' '}
            <span>
              {progression.loaded && progression.total
                ? `${((progression.loaded / progression.total) * 100).toFixed(2)}%`
                : null}
            </span>
          </div>
          <progress id="upload-progress" value={progression.loaded} max={progression.total} />
          {cancel !== undefined && (
            <div>
              <button
                className="btn btn-warning mt-1"
                onClick={() => {
                  cancel.abort();
                }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
