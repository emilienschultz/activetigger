import { FC, useState } from 'react';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import { useAppContext } from '../../../src/core/context';
import { useDeleteBertopic } from '../../core/api';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject },
  } = useAppContext();
  const deleteBertopic = useDeleteBertopic(projectName || null);
  const availableBertopic = currentProject ? currentProject.bertopic.available : [];
  const [currentBertopic, setCurrentBertopic] = useState<string | null>(null);

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explore">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <div className="explanations">
              Compute a Bertopic on the train dataset to identify the main topics
            </div>
            <h4 className="subsection">Existing Bertopic</h4>
            <div className="d-flex w-50">
              <Select
                className="flex-grow-1 "
                options={Object.keys(availableBertopic).map((e) => ({ value: e, label: e }))}
                onChange={(e) => {
                  if (e) setCurrentBertopic(e.value);
                }}
                value={{ value: currentBertopic, label: currentBertopic }}
              />
              <button
                className="btn btn p-0"
                onClick={() => {
                  deleteBertopic(currentBertopic);
                  setCurrentBertopic(null);
                }}
              >
                <MdOutlineDeleteOutline size={30} />
              </button>
            </div>
          </div>
          <div className="col-8">
            <h4 className="subsection">New Bertopic</h4>
            <BertopicForm projectSlug={projectName || null} />
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
