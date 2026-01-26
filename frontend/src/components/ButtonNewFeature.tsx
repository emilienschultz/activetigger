import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { useAppContext } from '../core/context';
import { CreateNewFeature } from './forms/CreateNewFeature';

export interface ButtonNewFeatureProps {
  projectSlug: string;
}

export const ButtonNewFeature: FC<ButtonNewFeatureProps> = ({ projectSlug }) => {
  const [displayNewFeature, setDisplayNewFeature] = useState(false);
  const {
    appContext: { currentProject },
  } = useAppContext();

  return (
    <>
      <button
        type="button"
        className="btn-secondary-action"
        style={{ flex: '1 1 auto' }}
        onClick={() => setDisplayNewFeature(true)}
      >
        <FaPlusCircle size={18} /> Add a new feature
      </button>
      <Modal
        show={displayNewFeature}
        id="features-modal"
        onHide={() => setDisplayNewFeature(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Add a new feature</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <CreateNewFeature
            projectName={projectSlug}
            featuresOption={currentProject?.features.options || {}}
            columns={currentProject?.params.all_columns || []}
            callback={() => setDisplayNewFeature(false)}
          />
        </Modal.Body>
      </Modal>
    </>
  );
};
