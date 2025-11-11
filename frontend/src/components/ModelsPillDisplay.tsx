import cx from 'classnames';
import { Dispatch, FC, ReactNode, SetStateAction, useState } from 'react';
import { Modal } from 'react-bootstrap';

import { MdOutlineDeleteOutline } from 'react-icons/md';

interface ModelsNameInput {
  modelNames: string[];
  currentModelName: string | null;
  setCurrentModelName: Dispatch<SetStateAction<string | null>>;
  deleteModelFunction: (model_name: string) => Promise<boolean | null | undefined | void> | void;
  children?: ReactNode;
}

export const ModelsPillDisplay: FC<ModelsNameInput> = ({
  modelNames,
  currentModelName,
  setCurrentModelName,
  deleteModelFunction,
  children,
}) => {
  const [showDelete, setShowDelete] = useState(false);
  return (
    <div className="model-pill-selection">
      {(modelNames || []).map((name) => (
        <button
          className={cx('model-pill ', currentModelName === name ? 'selected' : '')}
          onClick={() => {
            setCurrentModelName(name);
          }}
          key={name}
        >
          {name}
          <button
            id="bin"
            onClick={() => {
              setShowDelete(true);
            }}
          >
            <MdOutlineDeleteOutline size={20} />
          </button>
          <Modal show={showDelete} id={`deletescheme-${name}`} onHide={() => setShowDelete(false)}>
            <Modal.Header closeButton>
              <Modal.Title>Delete the current model</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <p>
                Are you sure you want to delete the model <b>{name}</b>?
              </p>
              <button
                className="btn btn-danger"
                onClick={() => {
                  deleteModelFunction(name);
                  if (currentModelName === name) {
                    setCurrentModelName(null);
                  }
                  setShowDelete(false);
                }}
              >
                Delete
              </button>
            </Modal.Body>
          </Modal>
        </button>
      ))}
      {children}
    </div>
  );
};
